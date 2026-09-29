"""Stage 1: fetch candidate AI news stories.

A source is anything with ``name`` and ``fetch() -> list[Story]``.
Register new ones in ``build_sources``.
"""
from __future__ import annotations

import html
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import Protocol

import requests

from .config import Config
from .models import Story

log = logging.getLogger(__name__)
UA = {"User-Agent": "ai-shorts/0.2 (+https://github.com/Nishanth-kalluri/Automated-AI-News)"}
AGENTMAIL_API = "https://api.agentmail.to/v0"
NEWSLETTER_MAX_CHARS = 24_000  # per issue; newsletters run 5-15k chars once links are inlined


class NewsSource(Protocol):
    name: str

    def fetch(self) -> list[Story]: ...


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


class RSSSource:
    name = "rss"

    def __init__(self, feeds: list[str]):
        self.feeds = feeds

    def fetch(self) -> list[Story]:
        import feedparser

        stories: list[Story] = []
        for url in self.feeds:
            try:
                resp = requests.get(url, headers=UA, timeout=15)
                resp.raise_for_status()
                feed = feedparser.parse(resp.content)
            except Exception as exc:  # one dead feed shouldn't stop the run
                log.warning("rss: %s failed: %s", url, exc)
                continue
            outlet = feed.feed.get("title", url)
            for e in feed.entries:
                ts = e.get("published_parsed") or e.get("updated_parsed")
                published = (
                    datetime(*ts[:6], tzinfo=timezone.utc) if ts else datetime.now(timezone.utc)
                )
                stories.append(
                    Story(
                        title=_clean(e.get("title", "")),
                        url=e.get("link", ""),
                        source=outlet,
                        published=published,
                        summary=_clean(e.get("summary", ""))[:600],
                    )
                )
        return stories


class HackerNewsSource:
    """AI-related HN stories with some traction, via the public Algolia API."""

    name = "hackernews"
    QUERIES = ["AI", "LLM", "OpenAI", "Anthropic", "Gemini", "machine learning"]

    def __init__(self, max_age_hours: float, min_points: int = 50):
        self.max_age_hours = max_age_hours
        self.min_points = min_points

    def fetch(self) -> list[Story]:
        since = int((datetime.now(timezone.utc) - timedelta(hours=self.max_age_hours)).timestamp())
        seen: dict[str, Story] = {}
        for q in self.QUERIES:
            try:
                resp = requests.get(
                    "https://hn.algolia.com/api/v1/search",
                    params={
                        "query": q,
                        "tags": "story",
                        "numericFilters": f"created_at_i>{since},points>{self.min_points}",
                        "hitsPerPage": 30,
                    },
                    headers=UA,
                    timeout=15,
                )
                resp.raise_for_status()
            except Exception as exc:
                log.warning("hackernews: query %r failed: %s", q, exc)
                continue
            for h in resp.json().get("hits", []):
                url = h.get("url") or f"https://news.ycombinator.com/item?id={h['objectID']}"
                seen.setdefault(
                    url,
                    Story(
                        title=h.get("title", ""),
                        url=url,
                        source="Hacker News",
                        published=datetime.fromtimestamp(h["created_at_i"], timezone.utc),
                        summary=f"{h.get('points', 0)} points, {h.get('num_comments', 0)} comments on HN",
                        popularity=min(h.get("points", 0) / 500, 1.0),
                    ),
                )
        return list(seen.values())


class _LinkText(HTMLParser):
    """HTML email -> readable text that keeps each link as "text (url)"."""

    SKIP = {"style", "script", "head", "title"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "table", "section"}

    def __init__(self):
        super().__init__()
        self.out: list[str] = []
        self.skip = 0
        self.href: str | None = None

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "a":
            self.href = dict(attrs).get("href")
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(self.skip - 1, 0)
        elif tag == "a":
            if self.href and self.href.startswith("http"):
                self.out.append(f" ({self.href})")
            self.href = None
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def html_to_text(markup: str) -> str:
    parser = _LinkText()
    parser.feed(markup or "")
    text = html.unescape("".join(parser.out))
    text = re.sub(r"[ \t\u00a0\u200c\u034f]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def _sender_name(sender: str) -> str:
    name = re.sub(r"<[^>]*>", "", sender or "").strip().strip('"')
    return name or sender


class NewsletterSource:
    """Reads the newsletter issues that arrived in the AgentMail inbox in the last N hours.

    Each issue becomes one Story of kind "newsletter" whose body is the issue's text with
    links kept inline; the editor stage reads these bodies to find the day's stories.
    """

    name = "newsletter"

    def __init__(self, api_key: str, inbox: str, max_age_hours: float):
        self.api_key, self.inbox, self.max_age_hours = api_key, inbox, max_age_hours

    def _get(self, path: str, **params) -> dict:
        resp = requests.get(f"{AGENTMAIL_API}/inboxes/{self.inbox}{path}", params=params,
                            headers={"Authorization": f"Bearer {self.api_key}", **UA}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def fetch(self) -> list[Story]:
        if not (self.api_key and self.inbox):
            log.warning("newsletter: AGENTMAIL_API_KEY or AGENTMAIL_INBOX not set; skipping")
            return []
        since = datetime.now(timezone.utc) - timedelta(hours=self.max_age_hours)
        stories: list[Story] = []
        try:
            listing = self._get("/messages", limit=50, after=since.isoformat())
        except Exception as exc:
            log.warning("newsletter: listing inbox failed: %s", exc)
            return []
        for item in listing.get("messages", []):
            if "sent" in item.get("labels", []) or self.inbox in item.get("from", ""):
                continue  # our own "episode ready" emails
            try:
                msg = self._get(f"/messages/{item['message_id']}")
            except Exception as exc:
                log.warning("newsletter: message %s failed: %s", item.get("message_id"), exc)
                continue
            body = html_to_text(msg["html"]) if msg.get("html") else (msg.get("text") or "")
            if len(body) < 400:
                continue  # confirmations, welcome mails
            ts = msg.get("timestamp") or item.get("timestamp")
            published = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else datetime.now(timezone.utc)
            stories.append(Story(
                title=msg.get("subject", "(no subject)"),
                url="",
                source=_sender_name(msg.get("from", "")),
                published=published,
                summary=(msg.get("preview") or "")[:300],
                kind="newsletter",
                body=body[:NEWSLETTER_MAX_CHARS],
            ))
        return stories


class RedditSource:
    """Top posts of the day from AI subreddits, via their public RSS feeds.

    Reddit's logged-out JSON endpoints return 403 since 2026, while the RSS feeds still answer
    if requests are spaced out. RSS has no scores, so posts get a flat popularity.
    """

    name = "reddit"
    SUBS = ["MachineLearning", "LocalLLaMA", "singularity", "OpenAI"]
    PAUSE_SECONDS = 2.0

    def fetch(self) -> list[Story]:
        import feedparser

        stories: list[Story] = []
        for i, sub in enumerate(self.SUBS):
            if i:
                time.sleep(self.PAUSE_SECONDS)
            try:
                resp = requests.get(f"https://www.reddit.com/r/{sub}/top/.rss", params={"t": "day", "limit": 15},
                                    headers=UA, timeout=15)
                resp.raise_for_status()
                feed = feedparser.parse(resp.content)
            except Exception as exc:
                log.warning("reddit: r/%s failed: %s", sub, exc)
                continue
            for e in feed.entries:
                content = " ".join(c.get("value", "") for c in e.get("content", [])) or e.get("summary", "")
                ts = e.get("published_parsed") or e.get("updated_parsed")
                stories.append(Story(
                    title=_clean(e.get("title", "")),
                    url=_reddit_link(content) or e.get("link", ""),
                    source=f"r/{sub}",
                    published=datetime(*ts[:6], tzinfo=timezone.utc) if ts else datetime.now(timezone.utc),
                    summary=_clean(re.sub(r"submitted by.*$", "", content, flags=re.S))[:400],
                    popularity=0.3,
                ))
        return stories


def _reddit_link(content: str) -> str:
    """The post's outbound link: the "[link]" anchor, unless it points back at Reddit."""
    m = re.search(r'<a href="([^"]+)">\s*\[link\]\s*</a>', content or "")
    url = html.unescape(m.group(1)) if m else ""
    return "" if "reddit.com" in url or "redd.it" in url else url


class SampleSource:
    """Offline fixture so the pipeline runs with no network at all."""

    name = "sample"

    def fetch(self) -> list[Story]:
        now = datetime.now(timezone.utc)
        items = [
            ("Open-weight model tops coding leaderboard", "A new open-weight language model matched closed frontier models on popular coding benchmarks while running on a single GPU node."),
            ("Major lab ships AI agent that books travel end to end", "The agent can search flights, compare hotels and complete checkout, asking the user to confirm before any payment."),
            ("EU publishes first guidance for general-purpose AI audits", "Regulators outlined how providers of large models should document training data and report serious incidents."),
            ("Chipmaker unveils inference accelerator claiming 3x efficiency", "The company says the chip cuts the energy cost of serving large language models by two thirds."),
            ("Researchers show small models can learn to use tools from a few examples", "A new paper finds that fine-tuning on a handful of tool-use traces transfers to unseen APIs."),
            ("Video model generates one minute clips with consistent characters", "The model keeps faces and outfits stable across scenes, a common weakness of earlier video generators."),
            ("Hospital network reports AI triage cut emergency wait times by 20 percent", "The system flags high-risk patients from intake notes so nurses see them sooner."),
            ("Startup raises 200 million dollars to build AI for chip design", "Its agents propose circuit layouts that engineers then verify, shortening design cycles."),
            ("Popular coding assistant adds voice mode", "Developers can now describe changes out loud and review the proposed edits before applying them."),
        ]
        return [
            Story(title=t, url=f"https://example.com/sample/{i}", source="Sample Wire",
                  published=now - timedelta(hours=i * 2), summary=s)
            for i, (t, s) in enumerate(items)
        ]


def build_sources(cfg: Config) -> list[NewsSource]:
    factories = {
        "newsletter": lambda: NewsletterSource(cfg.agentmail_api_key, cfg.agentmail_inbox, cfg.max_age_hours),
        "rss": lambda: RSSSource(cfg.rss_feeds),
        "hackernews": lambda: HackerNewsSource(cfg.max_age_hours),
        "reddit": RedditSource,
        "sample": SampleSource,
    }
    unknown = [n for n in cfg.sources if n not in factories]
    if unknown:
        raise ValueError(f"Unknown news source(s): {unknown}. Known: {sorted(factories)}")
    return [factories[n]() for n in cfg.sources]


def fetch_all(sources: list[NewsSource]) -> list[Story]:
    stories: list[Story] = []
    for src in sources:
        try:
            got = src.fetch()
        except Exception as exc:  # a broken source should cost us stories, not the run
            log.warning("%s: failed: %s", src.name, exc)
            got = []
        log.info("%s: %d stories", src.name, len(got))
        stories.extend(got)
    return stories
