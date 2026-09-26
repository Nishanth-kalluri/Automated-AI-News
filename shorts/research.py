"""Stage 3: read the full article behind each picked story.

The editor's summary comes from newsletter text; this stage adds the source article so the
writer has more facts to work with. Without a Tavily key it does nothing.
"""
from __future__ import annotations

import logging
from typing import Protocol

import requests

from .models import Story

log = logging.getLogger(__name__)
TAVILY_EXTRACT = "https://api.tavily.com/extract"
ARTICLE_MAX_CHARS = 6000
UA = {"User-Agent": "Mozilla/5.0 (compatible; ai-shorts/0.2)"}


class Researcher(Protocol):
    name: str

    def enrich(self, stories: list[Story]) -> None: ...


def resolve_url(url: str) -> str:
    """Follow newsletter tracking redirects to the real article URL."""
    try:
        resp = requests.head(url, allow_redirects=True, timeout=10, headers=UA)
        if resp.status_code >= 400:  # some sites refuse HEAD
            resp = requests.get(url, allow_redirects=True, timeout=10, headers=UA, stream=True)
            resp.close()
        return resp.url or url
    except Exception:
        return url


class TavilyResearcher:
    name = "tavily"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def enrich(self, stories: list[Story]) -> None:
        targets = [s for s in stories if s.url.startswith("http")]
        for s in targets:
            s.url = resolve_url(s.url)
        if not targets:
            return
        resp = requests.post(
            TAVILY_EXTRACT,
            json={"urls": [s.url for s in targets][:20], "extract_depth": "basic", "format": "text"},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()
        by_url = {r["url"]: r.get("raw_content") or "" for r in data.get("results", [])}
        for s in targets:
            text = by_url.get(s.url, "")
            if text:
                s.body = text[:ARTICLE_MAX_CHARS]
        failed = [f.get("url") for f in data.get("failed_results", [])]
        log.info("      read %d of %d articles%s", sum(1 for s in targets if s.body), len(targets),
                 f" (failed: {', '.join(failed)})" if failed else "")


class NoResearch:
    name = "none"

    def enrich(self, stories: list[Story]) -> None:
        return None


def build_researcher(tavily_api_key: str) -> Researcher:
    return TavilyResearcher(tavily_api_key) if tavily_api_key else NoResearch()


def research(researcher: Researcher, stories: list[Story]) -> None:
    try:
        researcher.enrich(stories)
    except Exception as exc:  # the newsletter summary is still enough to write from
        log.warning("%s research failed (%s); writing from summaries only", researcher.name, exc)
