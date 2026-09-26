"""One tiny interface over the LLM providers: send a prompt, get JSON back.

Add a provider by writing a class with ``name`` and ``json(system, user) -> dict``
and registering it in ``build_llm``.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

from .config import Config

log = logging.getLogger(__name__)


@dataclass
class Usage:
    """Token counts per call, written to cost.json so real spend is visible."""

    calls: list[dict] = field(default_factory=list)

    def add(self, stage: str, model: str, input_tokens: int, output_tokens: int) -> None:
        self.calls.append({"stage": stage, "model": model,
                           "input_tokens": input_tokens, "output_tokens": output_tokens})


class LLM(Protocol):
    name: str
    model: str

    def json(self, system: str, user: str, *, stage: str) -> dict: ...


def parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError(f"LLM did not return JSON: {text[:300]!r}")
        return json.loads(match.group(0))


class OpenAILLM:
    name = "openai"

    def __init__(self, api_key: str, model: str, usage: Usage):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, timeout=300, max_retries=3)
        self.model, self.usage = model, usage

    def json(self, system: str, user: str, *, stage: str) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
        )
        if resp.usage:
            self.usage.add(stage, self.model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return parse_json(resp.choices[0].message.content or "")


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, api_key: str, model: str, usage: Usage):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key, max_retries=3)
        self.model, self.usage = model, usage

    def json(self, system: str, user: str, *, stage: str) -> dict:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=16000,
            system=system + "\n\nReply with a single JSON object and nothing else.",
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
