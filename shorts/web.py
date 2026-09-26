"""Web tools for the research agents: Tavily search and extract, Jina Reader, and quote checks.

Every call is bounded. Tavily spends credits from a monthly counter kept in the state folder
(the free plan has 1,000 a month), Jina has a per-run call budget, and both switch themselves
off for the rest of the run after errors that won't go away (bad key, rate limit, used-up plan).
Callers get ``WebUnavailable`` or ``None``, never a crash.

The API shapes here are from the providers' docs as remembered, not checked from this sandbox;
anything unexpected degrades to "no result" (and a 400 retries once with the minimal payload).
"""
from __future__ import annotations

import calendar
import json
import logging
import math
import re
import threading
import unicodedata
from dataclasses import dataclass
from datetime import date
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

from .config import Config
from .coverage import Coverage
from .llm import write_json

log = logging.getLogger(__name__)
TAVILY_API = "https://api.tavily.com"
JINA_READER = "https://r.jina.ai/"
READ_CHARS = 6000
UA = {"User-Agent": "ai-shorts/0.4 (+https://github.com/Nishanth-kalluri/Automated-AI-News)"}
JUNK_MARKERS = ("subscribe to continue", "sign in to read", "enable javascript", "access denied",
                "are you a robot", "captcha", "target url returned error")


@dataclass
class Doc:
    url: str
    title: str = ""
    text: str = ""
    snippet: str = ""
    published: str = ""  # YYYY-MM-DD when known
    via: str = ""  # tavily, jina, search or feed


class CreditsExhausted(RuntimeError):
    pass


class WebUnavailable(RuntimeError):
    pass


def iso_date(value: str) -> str:
    """YYYY-MM-DD from an ISO or RFC 2822 timestamp, or "" if it can't be read."""
    value = (value or "").strip()
    if not value:
        return ""
    try:
        return date.fromisoformat(value[:10]).isoformat()
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(value).date().isoformat()
    except (TypeError, ValueError, IndexError):
        return ""


class TavilyCredits:
    """Tavily credits spent per calendar month, in ``state/tavily.json`` ({"2026-09": 214}).

    Each run may spend at most ``run_cap`` and at most its fair share of what is left this month,
    so a busy day can't use up the rest of the month. ``path=None`` counts in memory without caps.
    """

    def __init__(self, path: Path | None, month_cap: int = 700, run_cap: int = 25, today: date | None = None):
        self.path, self.month_cap, self.run_cap = path, month_cap, run_cap
        self.today = today or date.today()
        self._lock = threading.Lock()
        self.months: dict[str, int] = {}
        if path and path.exists():
            try:
                self.months = {k: int(v) for k, v in json.loads(path.read_text()).items()}
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                log.warning("tavily: can't read %s (%s); counting from 0", path, exc)
        self.run_used = 0
        self.allowance: float = math.inf
        self.new_run()

    @property
    def month(self) -> str:
        return self.today.strftime("%Y-%m")

    def this_month(self) -> int:
        return self.months.get(self.month, 0)

    def new_run(self) -> None:
        with self._lock:
            self.run_used = 0
            if self.path is None:
                self.allowance = math.inf
                return
            days_left = calendar.monthrange(self.today.year, self.today.month)[1] - self.today.day + 1
            left = max(self.month_cap - self.this_month(), 0)
            self.allowance = max(0, min(self.run_cap, left // days_left))

    @property
    def remaining(self) -> float:
        with self._lock:
            month_left = math.inf if self.path is None else self.month_cap - self.this_month()
            return max(min(self.allowance - self.run_used, month_left), 0)

    def charge(self, n: int) -> None:
        with self._lock:
            if self.run_used + n > self.allowance:
                raise CreditsExhausted(f"Tavily credit limit for this run reached ({self.run_used} used)")
            if self.path is not None and self.this_month() + n > self.month_cap:
                raise CreditsExhausted(f"Tavily credit limit for the month reached ({self.this_month()} used)")
            self.run_used += n
            self.months[self.month] = self.this_month() + n
            if self.path is not None:
                write_json(self.path, self.months)  # saved as it happens, like the spend ledger

    def exhaust(self) -> None:
        """Tavily said the plan is used up: skip it for the rest of the month."""
        with self._lock:
            self.months[self.month] = max(self.this_month(), self.month_cap)
            if self.path is not None:
                write_json(self.path, self.months)


class Tavily:
    """Tavily search (1 credit) and extract (1 credit per 5 links), counted and self-disabling."""

    SEARCH_MINIMAL = ("query", "max_results")
    EXTRACT_MINIMAL = ("urls",)

    def __init__(self, api_key: str, credits: TavilyCredits, timeout: float = 20):
        self.api_key, self.credits, self.timeout = api_key, credits, timeout
        self.disabled = ""  # why it is off for the rest of the run
        self.minimal = False  # a 400 once: send only the required fields from then on
        self._fails = 0
        self._lock = threading.Lock()

    @property
    def usable(self) -> bool:
        return bool(self.api_key) and not self.disabled and self.credits.remaining > 0

    def _off(self, reason: str) -> None:
        with self._lock:
            if not self.disabled:
                self.disabled = reason
                log.warning("tavily: %s; off for the rest of this run", reason)

    def _failed(self, reason: str) -> WebUnavailable:
        with self._lock:
            self._fails += 1
            twice = self._fails >= 2
        if twice:
            self._off(f"{reason}, twice in a row")
        return WebUnavailable(f"Tavily: {reason}")

    def _post(self, path: str, payload: dict, minimal: tuple[str, ...], credits: int) -> dict:
        if not self.api_key:
            raise WebUnavailable("no Tavily key")
        if self.disabled:
            raise WebUnavailable(f"Tavily is off for this run ({self.disabled})")
        try:
            self.credits.charge(credits)  # before the call, so a crash mid-call still counts
        except CreditsExhausted as exc:
            raise WebUnavailable(str(exc)) from None
        body = {k: v for k, v in payload.items() if k in minimal} if self.minimal else payload
        for _ in range(2):
            try:
                resp = requests.post(f"{TAVILY_API}/{path}", json=body, timeout=self.timeout,
                                     headers={"Authorization": f"Bearer {self.api_key}"})
            except requests.RequestException as exc:
                raise self._failed(type(exc).__name__) from None
            status = resp.status_code
            if status == 400 and len(body) > len(minimal):
                log.warning("tavily: %s rejected the request (400); retrying with the required fields only", path)
                self.minimal = True
                body = {k: v for k, v in payload.items() if k in minimal}
                continue
            if status in (401, 403, 429):
                self._off(f"HTTP {status}")
                raise WebUnavailable(f"Tavily: HTTP {status}")
            if status in (432, 433):
                self._off("the plan's credits are used up")
                self.credits.exhaust()
                raise WebUnavailable("Tavily: plan credits used up")
            if status >= 500:
                raise self._failed(f"HTTP {status}")
            if status >= 400:
                raise WebUnavailable(f"Tavily: HTTP {status}")
            with self._lock:
                self._fails = 0
            try:
                data = resp.json()
            except ValueError:
                return {}
            return data if isinstance(data, dict) else {}
        raise WebUnavailable("Tavily: request rejected")

    def search(self, query: str, max_results: int = 5) -> list[Doc]:
        data = self._post("search", {"query": query[:400], "search_depth": "basic", "topic": "general",
                                     "time_range": "week", "max_results": max_results,
                                     "include_raw_content": True}, self.SEARCH_MINIMAL, 1)
        docs = []
        for r in data.get("results") or []:
            if not isinstance(r, dict) or not str(r.get("url", "")).startswith("http"):
                continue
            content = str(r.get("content") or "")
            docs.append(Doc(url=r["url"], title=str(r.get("title") or ""),
                            text=str(r.get("raw_content") or content)[:READ_CHARS], snippet=content[:300],
                            published=iso_date(str(r.get("published_date") or "")), via="search"))
        return docs

    def extract(self, urls: list[str]) -> dict[str, Doc]:
        urls = [u for u in urls if u.startswith("http")][:20]
        if not urls:
            return {}
        data = self._post("extract", {"urls": urls, "extract_depth": "basic", "format": "text"},
                          self.EXTRACT_MINIMAL, math.ceil(len(urls) / 5))
        out = {}
        for r in data.get("results") or []:
            if isinstance(r, dict) and r.get("url") and r.get("raw_content"):
                out[r["url"]] = Doc(url=r["url"], text=str(r["raw_content"]), via="tavily")
        failed = [f.get("url") for f in data.get("failed_results") or [] if isinstance(f, dict)]
        if failed:
            log.info("      tavily could not read: %s", ", ".join(str(u) for u in failed))
        return out


class Jina:
    """Jina Reader (r.jina.ai): a page as plain text, free without a key at a low rate."""

    def __init__(self, timeout: float = 20, max_calls: int = 24):
        self.timeout, self.max_calls = timeout, max_calls
        self.calls = 0
        self.disabled = ""
        self._lock = threading.Lock()
        self._slots = threading.Semaphore(3)

    def read(self, url: str) -> Doc | None:
        if self.disabled or not url.startswith("http"):
            return None
        with self._lock:
            if self.calls >= self.max_calls:
                return None
            self.calls += 1
        with self._slots:
            try:
                resp = requests.get(JINA_READER + url, headers={**UA, "Accept": "text/plain"}, timeout=self.timeout)
            except requests.RequestException as exc:
                log.info("      jina: %s failed (%s)", url, type(exc).__name__)
                return None
        if resp.status_code == 429:
            with self._lock:
                if not self.disabled:
                    self.disabled = "rate limited"
                    log.warning("jina: rate limited; off for the rest of this run")
            return None
        if resp.status_code != 200:
            return None
        doc = parse_jina(url, resp.text or "")
        reason = junk_reason(doc.text)
        if reason:
            log.info("      jina: %s gave no article (%s)", url, reason)
            return None
        return doc


def parse_jina(url: str, body: str) -> Doc:
    """Jina's reply starts with "Title:", "URL Source:" and "Published Time:" lines, then
    "Markdown Content:" and the page. Without those lines the whole reply is the text."""
    head, sep, content = body.partition("Markdown Content:")
    fields = {}
    if sep:
        for line in head.splitlines():
            key, colon, value = line.partition(":")
            if colon:
                fields[key.strip().lower()] = value.strip()
    return Doc(url=fields.get("url source") or url, title=fields.get("title", ""),
               text=(content if sep else body).strip()[:READ_CHARS],
               published=iso_date(fields.get("published time", "")), via="jina")


def junk_reason(text: str) -> str | None:
    """Why a fetched page is not an article (paywall, bot check, error page), or None."""
    if len((text or "").strip()) < 500:
        return "too short"
    low = text.lower()
    return next((f"says {m!r}" for m in JUNK_MARKERS if m in low), None)


def normalize(text: str) -> str:
    """Comparable text: same quotes, dashes and spaces, no markdown links or emphasis."""
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = re.sub(r"[‘’‚‛′]", "'", text)
    text = re.sub(r"[“”„‟″]", '"', text)
    text = re.sub(r"[‐-―−]", "-", text)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [text](link) -> text
    text = re.sub(r"[*_`#]+", "", text)
    text = re.sub(r"\s+%", "%", text)  # "93.4 %" (often a narrow space) reads as "93.4%"
    return re.sub(r"\s+", " ", text).strip()


def quote_in(quote: str, text: str) -> bool:
    """Whether ``quote`` appears in ``text``. A quote with "..." must have every part of 4 or more
    words there, in order."""
    q, t = normalize(quote), normalize(text)
    if not q or not t:
        return False
    parts = [p.strip(" ,;:") for p in re.split(r"\.\.\.|…", q)]
    if len(parts) == 1:
        return len(q.split()) >= 4 and q in t
    parts = [p for p in parts if len(p.split()) >= 4]
    if not parts:
        return False
    pos = 0
    for p in parts:
        found = t.find(p, pos)
        if found < 0:
            return False
        pos = found + len(p)
    return True


@dataclass
class Web:
    tavily: Tavily | None
    jina: Jina
    coverage: Coverage


def build_web(cfg: Config, credits: TavilyCredits) -> Web | None:
    """The web tools for SHORTS_WEB=on runs on live news; None otherwise (no HTTP at all)."""
    if not cfg.web or cfg.sources == ["sample"]:
        return None
    tavily = Tavily(cfg.tavily_api_key, credits) if cfg.tavily_api_key else None
    return Web(tavily=tavily, jina=Jina(), coverage=Coverage(cfg.max_age_hours))
