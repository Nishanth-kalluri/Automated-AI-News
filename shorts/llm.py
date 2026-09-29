"""One small interface over the LLM providers.

``json(system, user, stage=..., schema=...)`` sends a prompt and returns a dict. OpenAI also
offers ``run_tools(...)``, a bounded tool-calling loop the agents use. Every call is priced
and counted against a per-run and a per-month dollar cap; once a cap is hit the next call
raises ``BudgetExceeded`` and the stage falls back to its no-LLM path.

Add a provider by writing a class with ``name``, ``model`` and ``json(...)`` and registering
it in ``build_llm``.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Protocol

from .config import Config

log = logging.getLogger(__name__)

# USD per 1M tokens: (input, cached input, output). Checked on OpenAI's pricing page on
# 2026-09-26 except gpt-5-mini (third-party table). Only an exact name or a dated snapshot
# ("gpt-5-2025-08-07") matches, so "gpt-5-pro" or "gpt-5.5" is not priced as gpt-5. Unknown
# models are priced high on purpose, so a missing entry can only make the budget cap trip early.
PRICES = {
    "gpt-6-sol": (2.00, 0.20, 10.00),
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-5": (1.25, 0.125, 10.00),
    "gpt-5-mini": (0.25, 0.025, 2.00),
}
UNKNOWN_PRICE = (10.00, 1.00, 50.00)  # the top of the regular tiers
PRO_PRICE = (150.00, 15.00, 600.00)  # "pro" tiers cost many times their base model
_warned_prices: set[str] = set()
TOOL_OUTPUT_MAX_CHARS = 8000
# What one parallel worker call (a researcher turn) is assumed to use, input and output tokens,
# held against the caps while it runs.
WORKER_CALL_TOKENS = (8000, 2000)


class BudgetExceeded(RuntimeError):
    pass


def price(model: str) -> tuple[float, float, float]:
    for name, p in PRICES.items():
        if model == name or re.fullmatch(re.escape(name) + r"-\d{4}-\d{2}-\d{2}", model):
            return p
    fallback = PRO_PRICE if re.search(r"-pro\b", model) else UNKNOWN_PRICE
    if model not in _warned_prices:
        _warned_prices.add(model)
        log.warning("no price listed for %s; counting it at $%g/$%g per 1M tokens against the caps", model,
                    fallback[0], fallback[2])
    return fallback


def call_estimate(model: str, input_tokens: int, output_tokens: int) -> float:
    p_in, _, p_out = price(model)
    return (input_tokens * p_in + output_tokens * p_out) / 1e6


@dataclass
class Usage:
    """Tokens and dollars per call, written to cost.json, with the run's spending caps."""

    calls: list[dict] = field(default_factory=list)
    run_cap_usd: float = float("inf")
    month_cap_usd: float = float("inf")
    month_spent_usd: float = 0.0  # spent by earlier runs this month
    # Called with each call's dollars as it happens, so the ledger is saved even if the run is killed.
    on_add: Callable[[float], None] | None = field(default=None, repr=False, compare=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)  # researchers run in threads
    _held_usd: float = field(default=0.0, repr=False, compare=False)  # estimates for calls still running

    def add(self, stage: str, model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> None:
        p_in, p_cached, p_out = price(model)
        usd = ((input_tokens - cached_tokens) * p_in + cached_tokens * p_cached + output_tokens * p_out) / 1e6
        with self._lock:
            self.calls.append({"stage": stage, "model": model, "input_tokens": input_tokens,
                               "cached_tokens": cached_tokens, "output_tokens": output_tokens, "usd": round(usd, 5)})
            if self.on_add:
                self.on_add(usd)

    @property
    def total_usd(self) -> float:
        with self._lock:
            return sum(c["usd"] for c in self.calls)

    def over_budget(self, reserve_usd: float = 0.0) -> bool:
        """``reserve_usd`` keeps that much of both caps back for later stages (the researchers leave
        room for the writer). Calls still running count at their estimate."""
        with self._lock:
            spent = sum(c["usd"] for c in self.calls) + self._held_usd + reserve_usd
        return spent >= self.run_cap_usd or self.month_spent_usd + spent >= self.month_cap_usd

    def check(self, stage: str, reserve_usd: float = 0.0) -> None:
        self.hold(stage, 0.0, reserve_usd)

    def hold(self, stage: str, estimate_usd: float, reserve_usd: float = 0.0) -> None:
        """Check the caps, counting calls still running, and hold ``estimate_usd`` for this call until
        ``release``. Parallel callers can't all pass on the same total."""
        with self._lock:
            total = sum(c["usd"] for c in self.calls)
            spent = total + self._held_usd + estimate_usd + reserve_usd
            if spent >= self.run_cap_usd or self.month_spent_usd + spent >= self.month_cap_usd:
                held = f", keeping ${reserve_usd:.2f} for later stages" if reserve_usd else ""
                raise BudgetExceeded(f"{stage}: spending cap reached (${total:.2f} this run, "
                                     f"${self.month_spent_usd + total:.2f} this month{held})")
            self._held_usd += estimate_usd

    def release(self, estimate_usd: float) -> None:
        with self._lock:
            self._held_usd = max(self._held_usd - estimate_usd, 0.0)


class SpendLedger:
    """Dollars spent per calendar month, kept in the state folder so the monthly cap spans runs."""

    def __init__(self, path: Path):
        self.path = path
        self.months: dict[str, float] = json.loads(path.read_text()) if path.exists() else {}

    @staticmethod
    def _month() -> str:
        return date.today().strftime("%Y-%m")

    def this_month(self) -> float:
        return self.months.get(self._month(), 0.0)

    def add(self, usd: float) -> None:
        if usd <= 0:
            return
        self.months[self._month()] = round(self.this_month() + usd, 5)
        write_json(self.path, self.months)


def write_json(path: Path, data) -> None:
    """Write through a temporary file, so a run killed mid-write never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


@dataclass
class Tool:
    """A function the model may call. ``parameters`` is a strict JSON schema for its arguments."""

    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]

    def spec(self) -> dict:
        return {"type": "function", "name": self.name, "description": self.description,
                "parameters": self.parameters, "strict": True}

    def call(self, arguments: str) -> str:
        try:
            result = self.fn(**json.loads(arguments or "{}"))
        except Exception as exc:  # the model sees the error and can try something else
            return f"error: {exc}"
        text = result if isinstance(result, str) else json.dumps(result, default=str)
        return text[:TOOL_OUTPUT_MAX_CHARS]


class LLM(Protocol):
    name: str
    model: str

    def json(self, system: str, user: str, *, stage: str, schema: dict | None = None) -> dict: ...


def parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError(f"LLM did not return JSON: {text[:300]!r}")
        return json.loads(match.group(0))


def strict_object(properties: dict, *, description: str = "") -> dict:
    """JSON schema object in the form OpenAI's strict mode needs: every field required, no extras."""
    schema = {"type": "object", "properties": properties, "required": list(properties),
              "additionalProperties": False}
    if description:
        schema["description"] = description
    return schema


def capped(tools: list[Tool], limit: int) -> list[Tool]:
    """The same tools sharing one call budget; calls past ``limit`` get an error instead of running."""
    lock, used = threading.Lock(), [0]

    def wrap(tool: Tool) -> Tool:
        def fn(**kwargs):
            with lock:
                used[0] += 1
                over = used[0] > limit
            if over:
                return "error: tool limit reached; answer with what you have"
            return tool.fn(**kwargs)
        return Tool(tool.name, tool.description, tool.parameters, fn)

    return [wrap(t) for t in tools]


def with_model(llm: LLM | None, model: str) -> LLM | None:
    """The same provider, usage and caps, but a different model (for per-role models)."""
    if llm is None or not model or model == llm.model or not hasattr(llm, "with_model"):
        return llm
    return llm.with_model(model)


class OpenAILLM:
    name = "openai"

    def __init__(self, api_key: str, model: str, usage: Usage, fallback_model: str = "", client=None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, timeout=300, max_retries=3)
        self.client, self.model, self.usage = client, model, usage
        self.fallback_model = fallback_model if fallback_model != model else ""
        self.reserve_usd = 0.0  # budget this instance leaves for later stages
        self.hold_tokens: tuple[int, int] | None = None  # set on worker copies

    def with_model(self, model: str) -> "OpenAILLM":
        return OpenAILLM("", model, self.usage, fallback_model=self.model, client=self.client)

    def worker(self, *, timeout: float = 45.0, max_retries: int = 1, reserve_usd: float = 0.0) -> "OpenAILLM":
        """A copy for one thread: same model, fallback and usage, its own fallback state, shorter timeouts."""
        client = self.client
        if hasattr(client, "with_options"):
            client = client.with_options(timeout=timeout, max_retries=max_retries)
        clone = OpenAILLM("", self.model, self.usage, fallback_model=self.fallback_model, client=client)
        clone.reserve_usd = reserve_usd
        clone.hold_tokens = WORKER_CALL_TOKENS  # parallel calls hold their likely cost while running
        return clone

    def _create(self, stage: str, **kwargs):
        """One Responses API call, priced and recorded. Falls back once if the model isn't available."""
        estimate = call_estimate(self.model, *self.hold_tokens) if self.hold_tokens else 0.0
        self.usage.hold(stage, estimate, self.reserve_usd)
        try:
            try:
                resp = self.client.responses.create(model=self.model, **kwargs)
            except Exception as exc:
                if not (self.fallback_model and _is_model_error(exc)):
                    raise
                log.warning("%s: model %s unavailable (%s); using %s", stage, self.model, exc, self.fallback_model)
                self.model, self.fallback_model = self.fallback_model, ""
                resp = self.client.responses.create(model=self.model, **kwargs)
            u = getattr(resp, "usage", None)
            if u is not None:
                details = getattr(u, "input_tokens_details", None)
                cached = getattr(details, "cached_tokens", 0) or 0
                self.usage.add(stage, self.model, u.input_tokens, u.output_tokens, cached)
        finally:
            self.usage.release(estimate)
        return resp

    @staticmethod
    def _text_format(schema: dict | None, stage: str) -> dict:
        if schema:
            return {"format": {"type": "json_schema", "name": re.sub(r"\W", "_", stage)[:64],
                               "schema": schema, "strict": True}}
        return {"format": {"type": "json_object"}}

    def json(self, system: str, user: str, *, stage: str, schema: dict | None = None) -> dict:
        resp = self._create(stage, instructions=system, input=user, text=self._text_format(schema, stage))
        return parse_json(resp.output_text or "")

    def run_tools(self, system: str, user: str, tools: list[Tool], *, stage: str, schema: dict | None = None,
                  max_turns: int = 6, deadline: float | None = None) -> dict:
        """Let the model call ``tools`` for up to ``max_turns`` turns, then return its JSON answer.

        On the last turn, or once ``time.monotonic()`` passes ``deadline``, tools are switched off,
        so the loop always ends with an answer.
        """
        by_name = {t.name: t for t in tools}
        specs = [t.spec() for t in tools]
        text = self._text_format(schema, stage)
        resp = self._create(stage, instructions=system, input=user, tools=specs, text=text)
        for turn in range(1, max_turns + 1):
            calls = [item for item in resp.output if getattr(item, "type", "") == "function_call"]
            if not calls:
                return parse_json(resp.output_text or "")
            outputs = []
            for c in calls:
                tool = by_name.get(c.name)
                result = tool.call(c.arguments) if tool else f"error: no tool named {c.name}"
                log.info("      %s -> %s(%s)", stage, c.name, (c.arguments or "")[:120])
                outputs.append({"type": "function_call_output", "call_id": c.call_id, "output": result})
            last = turn == max_turns or (deadline is not None and time.monotonic() >= deadline)
            resp = self._create(stage, instructions=system, input=outputs, previous_response_id=resp.id,
                                tools=specs, tool_choice="none" if last else "auto", text=text)
            if last:
                break
        return parse_json(resp.output_text or "")


def _is_model_error(exc: Exception) -> bool:
    """The model doesn't exist or this key may not use it (a restricted project key answers 403)."""
    if getattr(exc, "code", None) == "model_not_found":
        return True
    return getattr(exc, "status_code", None) in (400, 403, 404) and "model" in str(exc).lower()


class AnthropicLLM:
    """JSON calls only; the tool-calling agents run on OpenAI and fall back to plain calls here."""

    name = "anthropic"

    def __init__(self, api_key: str, model: str, usage: Usage):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key, max_retries=3)
        self.model, self.usage = model, usage

    def with_model(self, model: str) -> "AnthropicLLM":
        clone = copy.copy(self)
        clone.model = model
        return clone

    def json(self, system: str, user: str, *, stage: str, schema: dict | None = None) -> dict:
        self.usage.check(stage)
        shape = f"\n\nThe JSON must match this schema:\n{json.dumps(schema)}" if schema else ""
        with self.client.messages.stream(
            model=self.model,
            max_tokens=16000,
            system=system + "\n\nReply with a single JSON object and nothing else." + shape,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            msg = stream.get_final_message()
        self.usage.add(stage, self.model, msg.usage.input_tokens, msg.usage.output_tokens)
        return parse_json("".join(b.text for b in msg.content if b.type == "text"))


def build_llm(cfg: Config, usage: Usage) -> LLM | None:
    """Returns None when no provider is configured; callers then use their template fallback."""
    try:
        if cfg.llm_provider == "openai" and cfg.openai_api_key:
            return OpenAILLM(cfg.openai_api_key, cfg.llm_model, usage)
        if cfg.llm_provider == "anthropic" and cfg.anthropic_api_key:
            return AnthropicLLM(cfg.anthropic_api_key, cfg.llm_model, usage)
    except ImportError as exc:
        log.warning("LLM package missing (%s); using templates", exc)
        return None
    if cfg.llm_provider not in ("none", "openai", "anthropic"):
        raise ValueError(f"Unknown LLM provider {cfg.llm_provider!r}")
    if cfg.llm_provider != "none":
        log.warning("SHORTS_LLM_PROVIDER=%s but its API key is missing; using templates", cfg.llm_provider)
    return None

