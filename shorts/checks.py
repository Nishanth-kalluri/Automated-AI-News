"""Rule checks shared by the pipeline and the agents.

Every check is a plain function that returns a list of ``Issue``s. The pipeline uses them as
guards, and the agent loops use the same issues to decide what to send back for repair, so
a problem is described once and fixed by whichever stage can fix it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .models import Episode, Story

SAMPLE_URL_PREFIX = "https://example.com/sample/"
STORY_WORDS = (30, 42)  # ~15 seconds of speech per story segment
WORDS_PER_SECOND = 2.6  # same pace the silent voice uses
SEGMENT_GAP = 0.3
TARGET_MAX_SECONDS = 170  # leaves headroom under the 180 s Shorts limit
HYPE_WORDS = ("revolutionary", "game-changer", "game changer", "mind-blowing", "mind blowing")
TITLE_MAX = 91  # the uploader appends " #Shorts" and YouTube allows 100 characters

_URL_RE = re.compile(r"https?://[^\s)>\]\"'<]+")
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


@dataclass
class Issue:
    code: str
    detail: str
    index: int | None = None  # 0-based story position, or None for the whole pick list / episode
    fatal: bool = True  # False: worth fixing, fine to ship if it can't be

    def line(self) -> str:
        where = f"story {self.index + 1}: " if self.index is not None else ""
        return f"- {where}{self.detail}"


def is_sample(story: Story) -> bool:
    return story.url.startswith(SAMPLE_URL_PREFIX)


def norm_url(url: str) -> str:
    """Comparable form of a URL: lower-case host, no tracking params, fragment or trailing slash."""
    url = (url or "").strip()
    if not url.startswith("http"):
        return ""
    parts = urlsplit(url)
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if not k.lower().startswith("utm_")])
    path = parts.path.rstrip("/")
    return urlunsplit(("https", parts.netloc.lower().removeprefix("www."), path, query, ""))


def _norm_title(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (text or "").lower())


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm_title(a), _norm_title(b)).ratio()


def source_urls(candidates: list[Story]) -> set[str]:
    """Every URL the day's material mentions: feed links plus links inside newsletter text."""
    urls = {norm_url(c.url) for c in candidates if c.url}
    for c in candidates:
        urls |= {norm_url(u.rstrip(".,;:")) for u in _URL_RE.findall(f"{c.body} {c.summary}")}
    urls.discard("")
    return urls


def check_picks(picks: list[Story], candidates: list[Story], n: int, seen_urls: set[str],
                aired_headlines: list[str], max_age_hours: float,
                now: datetime | None = None) -> list[Issue]:
    now = now or datetime.now(timezone.utc)
    known_urls = source_urls(candidates)
    aired_urls = {norm_url(u) for u in seen_urls} - {""}
    live_candidates = any(not is_sample(c) for c in candidates)
    issues: list[Issue] = []
    for i, s in enumerate(picks):
        name = s.headline or s.title
        if not (name and s.summary):
            issues.append(Issue("incomplete", "missing a headline or summary", i))
            continue
        url = norm_url(s.url)
        for j, other in enumerate(picks[:i]):
            if (url and url == norm_url(other.url)) or similar(name, other.headline or other.title) > 0.6:
                issues.append(Issue("duplicate", f"same event as story {j + 1} ({other.headline or other.title!r})", i))
                break
        if url and url in aired_urls:
            issues.append(Issue("already_aired", f"{name!r} already aired (same link)", i))
        else:
            match = next((h for h in aired_headlines if similar(name, h) > 0.7), None)
            if match:
                issues.append(Issue("already_aired", f"{name!r} looks like {match!r}, which already aired", i))
        if not url:
            issues.append(Issue("no_url", f"{name!r} has no link", i, fatal=False))
        elif url not in known_urls:
            issues.append(Issue("url_not_in_sources", f"the link {s.url} does not appear in today's material", i))
        age_h = (now - s.published).total_seconds() / 3600
        if age_h > max_age_hours:
            issues.append(Issue("stale", f"{name!r} was published {age_h:.0f} hours ago", i))
        if live_candidates and is_sample(s):
            issues.append(Issue("sample", f"{name!r} is a built-in sample story, not real news", i))
    if len(picks) < n:
        issues.append(Issue("too_few", f"only {len(picks)} of {n} stories"))
    return issues


def _numbers(text: str) -> set[str]:
    return {m.replace(",", "").rstrip(".") for m in _NUM_RE.findall(text or "")}


def predicted_seconds(episode: Episode) -> float:
    words = sum(len(s.text.split()) for s in episode.segments)
    return words / WORDS_PER_SECOND + SEGMENT_GAP * max(len(episode.segments) - 1, 0)


def lint_episode(episode: Episode, stories: list[Story]) -> list[Issue]:
    """Script rules the writer is told about, checked in code rather than trusted."""
    issues: list[Issue] = []
    kinds = [s.kind for s in episode.segments]
    if "intro" not in kinds or not episode.segments[0].text.strip():
        issues.append(Issue("missing_intro", "the intro is missing"))
    if "outro" not in kinds or not episode.segments[-1].text.strip():
        issues.append(Issue("missing_outro", "the outro is missing"))
    segs = episode.story_segments
    if len(segs) != len(stories):
        issues.append(Issue("segment_count", f"{len(segs)} story segments for {len(stories)} stories"))
    for i, (seg, story) in enumerate(zip(segs, stories)):
        words = len(seg.text.split())
        if words > STORY_WORDS[1] + 6:
            issues.append(Issue("too_long", f"{words} words; keep it to {STORY_WORDS[0]}-{STORY_WORDS[1]}", i))
        elif words < STORY_WORDS[0] - 8:
            issues.append(Issue("too_short", f"only {words} words; aim for {STORY_WORDS[0]}-{STORY_WORDS[1]}", i,
                                fatal=False))
        lowered = seg.text.lower()
        hype = [w for w in HYPE_WORDS if w in lowered]
        if hype:
            issues.append(Issue("hype", f"uses hype words ({', '.join(hype)})", i))
        if _URL_RE.search(seg.text) or re.search(r"\bwww\.", lowered):
            issues.append(Issue("url_in_narration", "reads out a web address", i))
        material = " ".join([story.title, story.headline, story.summary, story.body, story.key_fact])
        missing = sorted((_numbers(seg.text) | _numbers(seg.key_fact)) - _numbers(material))
        if missing:
            issues.append(Issue("unsupported_number",
                                f"uses {', '.join(missing)}, which is not in the story material", i))
        if len(seg.headline.split()) > 8:
            issues.append(Issue("long_headline", "headline is longer than 8 words", i, fatal=False))
    if len(episode.title) > TITLE_MAX:
        issues.append(Issue("long_title", f"title is {len(episode.title)} characters; keep it under {TITLE_MAX}",
                            fatal=False))
    seconds = predicted_seconds(episode)
    if seconds > TARGET_MAX_SECONDS:
        extra = int((seconds - TARGET_MAX_SECONDS) * WORDS_PER_SECOND) + 1
        issues.append(Issue("too_long_total", f"the episode runs about {seconds:.0f}s; cut about {extra} words"))
    return issues


def fatal(issues: list[Issue]) -> list[Issue]:
    return [i for i in issues if i.fatal]
