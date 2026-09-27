"""How widely a story is covered right now: Hacker News points and a Google News outlet count.

This is the "what's trending" signal for the editor. Both backends are free and keyless, and
either can be down, rate-limited or change format; then the answer is ``None`` ("unknown"),
never 0, so the editor is not misled into thinking a story is small.
"""
from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone

import requests

from .checks import _event_words, norm_url, same_event, similar
from .sources import UA, hn_search

log = logging.getLogger(__name__)
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
WINDOW_HOURS = 48


def event_query(headline: str, max_words: int = 5) -> str:
    """A short search query: the headline's event words, in their original order."""
    words = [w for w in re.findall(r"[\w$.'-]+", headline or "") if _event_words(w)]
    return " ".join(words[:max_words])


class Coverage:
    def __init__(self, max_age_hours: float = 30, max_requests: int = 40, now: datetime | None = None):
        self.max_requests = max_requests
        self.now = now or datetime.now(timezone.utc)
        self.since = self.now - timedelta(hours=max(WINDOW_HOURS, max_age_hours))
        self.requests = {"hn": 0, "news": 0}
        self.off: dict[str, str] = {}  # backend -> why it is off for this run
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _may_call(self, backend: str) -> bool:
        with self._lock:
            if backend in self.off or self.requests[backend] >= self.max_requests:
                return False
            self.requests[backend] += 1
            return True

    def _turn_off(self, backend: str, reason: str) -> None:
        with self._lock:
            if backend not in self.off:
                self.off[backend] = reason
                log.warning("coverage: %s %s; off for the rest of this run", backend, reason)

    def lookup(self, headline: str, url: str = "") -> dict:
        """{"hn_points", "hn_threads", "news_outlets", "outlets"}; a None count means unknown."""
        query = event_query(headline)
        key = query.lower()
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        result = {"headline": headline, "hn_points": None, "hn_threads": None, "news_outlets": None,
                  "outlets": []}
        if query:
            result.update(self._hacker_news(headline, url, query))
            result.update(self._google_news(headline, query))
        with self._lock:
            self._cache[key] = result
        return result

    def _hacker_news(self, headline: str, url: str, query: str) -> dict:
        if not self._may_call("hn"):
            return {}
        try:
            hits = hn_search(query, self.since, min_points=0, hits=20)
        except Exception as exc:
            log.info("      coverage: hacker news search failed (%s)", exc)
            return {}
        link = norm_url(url)
        matched = [h for h in hits if same_event(headline, h.get("title", ""))
                   or similar(headline, h.get("title", "")) >= 0.3 or (link and norm_url(h.get("url") or "") == link)]
        return {"hn_points": max((int(h.get("points") or 0) for h in matched), default=0), "hn_threads": len(matched)}

    def _google_news(self, headline: str, query: str) -> dict:
        if not self._may_call("news"):
            return {}
        try:
            resp = requests.get(GOOGLE_NEWS_RSS, params={"q": f"{query} when:2d", "hl": "en-US", "gl": "US",
                                                         "ceid": "US:en"}, headers=UA, timeout=10)
        except requests.RequestException as exc:
            log.info("      coverage: google news failed (%s)", type(exc).__name__)
            return {}
        if resp.status_code in (429, 503):
            self._turn_off("news", f"HTTP {resp.status_code}")
            return {}
        head = (resp.content or b"")[:500].lower()
        if resp.status_code != 200 or not (b"<rss" in head or b"<feed" in head):
            self._turn_off("news", "did not return a news feed")
            return {}
        import feedparser

        outlets: list[str] = []
        for e in feedparser.parse(resp.content).entries:
            title = e.get("title", "")
            source = (e.get("source") or {}).get("title", "")
            if not source and " - " in title:
                title, source = title.rsplit(" - ", 1)
            elif source and title.endswith(f" - {source}"):
                title = title[: -len(source) - 3]
            ts = e.get("published_parsed")
            if ts and datetime(*ts[:6], tzinfo=timezone.utc) < self.now - timedelta(hours=WINDOW_HOURS):
                continue
            if source and source not in outlets and (same_event(headline, title) or similar(headline, title) >= 0.25):
                outlets.append(source)
        return {"news_outlets": len(outlets), "outlets": outlets[:6]}

    def lookup_many(self, headlines: list[str], timeout: float = 30) -> dict[str, dict]:
        """Lookups in parallel; anything not back within ``timeout`` seconds is unknown."""
        pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="coverage")
        futures = {h: pool.submit(self.lookup, h) for h in headlines}
        wait(list(futures.values()), timeout=timeout)
        pool.shutdown(wait=False, cancel_futures=True)
        unknown = {"hn_points": None, "hn_threads": None, "news_outlets": None, "outlets": []}
        return {h: f.result() if f.done() and not f.cancelled() and f.exception() is None
                else {"headline": h, **unknown} for h, f in futures.items()}

    def rows(self) -> list[dict]:
        with self._lock:
            return list(self._cache.values())

    @staticmethod
    def line(r: dict) -> str:
        hn = "HN unknown" if r.get("hn_points") is None else f"HN {r['hn_points']} pts"
        news = ("Google News unknown" if r.get("news_outlets") is None
                else f"{r['news_outlets']} outlets on Google News")
        return f"{hn}, {news}"
