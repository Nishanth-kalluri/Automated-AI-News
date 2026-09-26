"""Stage 2: the editor picks today's stories.

``LLMEditor`` reads the newsletter issues plus feed headlines, merges coverage of the same
story, and picks the most important ones. ``HeuristicEditor`` is the no-key fallback that
scores feed headlines on AI relevance, freshness and popularity.
"""
from __future__ import annotations

import json
import logging
import math
import re
from datetime import date, datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Protocol

from .llm import LLM
from .models import Story

log = logging.getLogger(__name__)

AI_TERMS = {
    "ai": 1.0, "llm": 1.0, "gpt": 1.0, "claude": 1.0, "gemini": 1.0, "llama": 0.8,
    "openai": 1.0, "anthropic": 1.0, "deepmind": 1.0, "mistral": 0.8, "nvidia": 0.6,
    "model": 0.5, "agent": 0.7, "agents": 0.7, "chatbot": 0.6, "neural": 0.5,
    "machine learning": 0.8, "artificial intelligence": 1.0, "inference": 0.4,
    "benchmark": 0.4, "open-weight": 0.8, "open source": 0.4, "regulation": 0.4,
}
MAX_HEADLINES_FOR_LLM = 80


def _norm(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", title.lower())


def relevance(story: Story) -> float:
    text = f" {_norm(story.title)} {_norm(story.summary)} "
    hits = sum(w for term, w in AI_TERMS.items() if f" {term} " in text)
    return min(hits / 2.0, 1.0)


def score(story: Story, now: datetime, max_age_hours: float) -> float:
    age_h = max((now - story.published).total_seconds() / 3600, 0)
    if age_h > max_age_hours:
        return 0.0
    freshness = math.exp(-age_h / (max_age_hours / 2))
    return 0.5 * relevance(story) + 0.35 * freshness + 0.15 * story.popularity


class SeenStore:
    """Remembers what already aired so consecutive runs don't repeat stories.

    Stored as a list of {"date", "headline", "url"}; older files that are a plain list of
    URLs are still read.
    """

    KEEP_DAYS = 14

    def __init__(self, path: Path):
        self.path = path
        raw = json.loads(path.read_text()) if path.exists() else []
        self.entries: list[dict] = [e if isinstance(e, dict) else {"date": "", "headline": "", "url": e}
                                    for e in raw]

    @property
    def urls(self) -> set[str]:
        return {e["url"] for e in self.entries if e.get("url")}

    def recent_headlines(self, days: int = 7) -> list[str]:
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        return [e["headline"] for e in self.entries if e.get("headline") and e.get("date", "") >= cutoff]

    def add(self, stories: list[Story]) -> None:
        today = date.today().isoformat()
        self.entries += [{"date": today, "headline": s.headline or s.title, "url": s.url} for s in stories]
        cutoff = (date.today() - timedelta(days=self.KEEP_DAYS)).isoformat()
        self.entries = [e for e in self.entries if not e.get("date") or e["date"] >= cutoff]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=1))


def pick_stories(stories: list[Story], n: int, max_age_hours: float,
                 seen: set[str] | frozenset[str] = frozenset()) -> list[Story]:
    now = datetime.now(timezone.utc)
    for s in stories:
        s.score = score(s, now, max_age_hours)
    ranked = sorted((s for s in stories if s.score > 0 and s.url not in seen and s.title
                     and s.kind == "article"),
                    key=lambda s: s.score, reverse=True)
    chosen: list[Story] = []
    for s in ranked:
        if any(SequenceMatcher(None, _norm(s.title), _norm(c.title)).ratio() > 0.6 for c in chosen):
            continue  # same story from another outlet
        chosen.append(s)
        if len(chosen) == n:
            break
    return chosen


class Editor(Protocol):
    name: str

    def pick(self, candidates: list[Story], n: int, seen: SeenStore) -> list[Story]: ...


class HeuristicEditor:
    name = "heuristic"

    def __init__(self, max_age_hours: float):
        self.max_age_hours = max_age_hours

    def pick(self, candidates: list[Story], n: int, seen: SeenStore) -> list[Story]:
        picked = pick_stories(candidates, n, self.max_age_hours, seen.urls)
        for s in picked:
            s.headline = s.headline or s.title
            s.outlets = s.outlets or [s.source]
        return picked


EDITOR_SYSTEM = """You are the news editor of a daily 2 minute YouTube Short about AI news.
You read today's AI newsletters and feed headlines and decide which stories make the show.

How to choose:
- A story is one real-world event: a launch, a release, a paper, a funding round, a policy, a notable result.
- Merge coverage of the same event from different sources into one story and list every outlet.
- Rank by how much it matters to people who follow AI, and by how many independent sources covered it.
- Skip sponsored sections, ads, job posts, tutorials, prompt tips, polls, memes and "tools of the day" lists.
- Skip anything in the "already covered" list unless there is genuinely new information.
- Use only facts present in the texts. Never invent numbers, names or quotes.

Return JSON: {"stories": [ ... ]} with exactly the requested number of stories, most important first.
Each story: {
  "headline": "on-screen headline, at most 8 words",
  "summary": "3 to 5 sentences with the concrete facts from the texts: who, what, numbers, why it matters",
  "key_fact": "the single most striking number or fact, at most 7 words, or empty string",
  "url": "the best link for the story found in the texts, preferring the original announcement or article over newsletter links; empty string if none",
  "outlets": ["names of every newsletter or site that covered it"],
  "why": "one line on why it made the cut"
}"""


class LLMEditor:
    name = "llm"

    def __init__(self, llm: LLM, max_age_hours: float):
        self.llm, self.max_age_hours = llm, max_age_hours

    def _prompt(self, candidates: list[Story], n: int, seen: SeenStore) -> str:
        newsletters = [s for s in candidates if s.kind == "newsletter"]
        now = datetime.now(timezone.utc)
        articles = sorted((s for s in candidates if s.kind == "article" and s.title),
                          key=lambda s: score(s, now, self.max_age_hours), reverse=True)
        articles = [s for s in articles if s.url not in seen.urls][:MAX_HEADLINES_FOR_LLM]
        parts = [f"Today is {date.today().isoformat()}. Pick exactly {n} stories.\n"]
        for i, s in enumerate(newsletters, 1):
            parts.append(f"=== NEWSLETTER {i}: {s.source} | {s.title} | {s.published:%Y-%m-%d %H:%M} UTC ===\n{s.body}\n")
        if articles:
            parts.append("=== FEED HEADLINES (source | published | title | url | snippet) ===")
            parts += [f"- {s.source} | {s.published:%m-%d %H:%M} | {s.title} | {s.url} | {s.summary[:200]}"
                      for s in articles]
        recent = seen.recent_headlines()
        if recent:
            parts.append("\n=== ALREADY COVERED IN RECENT EPISODES ===")
            parts += [f"- {h}" for h in recent]
        return "\n".join(parts)

    def pick(self, candidates: list[Story], n: int, seen: SeenStore) -> list[Story]:
        data = self.llm.json(EDITOR_SYSTEM, self._prompt(candidates, n, seen), stage="editor")
        now = datetime.now(timezone.utc)
        picked = []
        for item in data.get("stories", [])[:n]:
            outlets = [o for o in item.get("outlets", []) if o] or ["AI newsletters"]
            picked.append(Story(
                title=item.get("headline", "").strip(),
                url=(item.get("url") or "").strip(),
                source=outlets[0],
                published=now,
                summary=item.get("summary", "").strip(),
                headline=item.get("headline", "").strip(),
                key_fact=item.get("key_fact", "").strip(),
                outlets=outlets,
            ))
        picked = [s for s in picked if s.headline and s.summary]
        if len(picked) < n:
            log.warning("editor returned %d of %d stories", len(picked), n)
        return picked


def build_editor(llm: LLM | None, max_age_hours: float) -> Editor:
    return LLMEditor(llm, max_age_hours) if llm else HeuristicEditor(max_age_hours)


def pick_with_fallback(editor: Editor, candidates: list[Story], n: int, seen: SeenStore,
                       max_age_hours: float) -> list[Story]:
    if isinstance(editor, HeuristicEditor):
        return editor.pick(candidates, n, seen)
    try:
        picked = editor.pick(candidates, n, seen)
        if picked:
            return picked
        log.warning("%s editor picked nothing; falling back to heuristic", editor.name)
    except Exception as exc:
        log.warning("%s editor failed (%s); falling back to heuristic", editor.name, exc)
    return HeuristicEditor(max_age_hours).pick(candidates, n, seen)
