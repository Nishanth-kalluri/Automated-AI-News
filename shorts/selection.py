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

from .checks import check_picks, fatal, norm_url, similar
from .llm import LLM, BudgetExceeded, Tool, strict_object
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
Every url must be copied exactly from the texts; never build or guess a link.
Each story: {
  "headline": "on-screen headline, at most 8 words",
  "summary": "3 to 5 sentences with the concrete facts from the texts: who, what, numbers, why it matters",
  "key_fact": "the single most striking number or fact, at most 7 words, or empty string",
  "url": "the best link for the story found in the texts, preferring the original announcement or article over newsletter links; empty string if none",
  "outlets": ["names of every newsletter or site that covered it"],
  "why": "one line on why it made the cut"
}"""


PICK_SCHEMA = strict_object({
    "headline": {"type": "string"},
    "summary": {"type": "string"},
    "key_fact": {"type": "string"},
    "url": {"type": "string"},
    "outlets": {"type": "array", "items": {"type": "string"}},
    "why": {"type": "string"},
})
EDITOR_SCHEMA = strict_object({"stories": {"type": "array", "items": PICK_SCHEMA}})


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
        data = self.llm.json(EDITOR_SYSTEM, self._prompt(candidates, n, seen), stage="editor", schema=EDITOR_SCHEMA)
        return self.to_stories(data, candidates, n)

    @staticmethod
    def to_stories(data: dict, candidates: list[Story], n: int) -> list[Story]:
        """Editor JSON -> Stories. A link that matches a feed article takes its date and publisher."""
        now = datetime.now(timezone.utc)
        articles = {norm_url(c.url): c for c in candidates if c.kind == "article" and c.url}
        picked = []
        for item in data.get("stories", [])[:n]:
            outlets = [o for o in item.get("outlets", []) if o] or ["AI newsletters"]
            url = (item.get("url") or "").strip()
            match = articles.get(norm_url(url))
            if match and match.source not in outlets:
                outlets.append(match.source)
            picked.append(Story(
                title=item.get("headline", "").strip(),
                url=url,
                source=match.source if match else outlets[0],
                published=match.published if match else now,
                summary=item.get("summary", "").strip(),
                popularity=match.popularity if match else 0.0,
                headline=item.get("headline", "").strip(),
                key_fact=item.get("key_fact", "").strip(),
                outlets=outlets,
            ))
        picked = [s for s in picked if s.headline and s.summary]
        if len(picked) < n:
            log.warning("editor returned %d of %d stories", len(picked), n)
        return picked


AGENT_TOOLS_NOTE = """

You have two tools. Use them before you answer; a few calls are enough.
- search_candidates(query): searches today's newsletters and feeds. Use it to find every outlet that covered an
  event (more outlets means a bigger story) and the best link for it. A link must come from today's material.
- aired_lookup(headline): checks whether a story already aired in the last two weeks."""

REPAIR_PROMPT = """

=== YOUR PICKS SO FAR ===
{picks}

=== PROBLEMS TO FIX ===
{problems}

Return the full list of {n} stories again. Keep the good ones exactly as they are and replace or fix only the
ones with problems. A replacement must be a different real event from the texts above."""


def _snippet(text: str, terms: list[str], width: int = 320) -> str:
    low = text.lower()
    pos = min((low.find(t) for t in terms if t in low), default=0)
    start = max(pos - width // 3, 0)
    return " ".join(text[start:start + width].split())


def candidate_search(candidates: list[Story], query: str, limit: int = 8) -> dict:
    terms = [t for t in re.findall(r"[a-z0-9][a-z0-9.+-]*", query.lower()) if len(t) > 1]
    if not terms:
        return {"matches": [], "outlets": 0}
    hits = [c for c in candidates if all(t in f"{c.title} {c.summary} {c.body}".lower() for t in terms)]
    rows = [{"source": c.source, "kind": c.kind, "title": c.title, "url": c.url,
             "published": c.published.strftime("%Y-%m-%d %H:%M"), "popularity": round(c.popularity, 2),
             "snippet": _snippet(f"{c.summary} {c.body}", terms)} for c in hits[:limit]]
    return {"matches": rows, "outlets": len({c.source for c in hits})}


def aired_search(seen: SeenStore, headline: str) -> dict:
    scored = sorted(((similar(headline, e.get("headline", "")), e) for e in seen.entries if e.get("headline")),
                    key=lambda x: x[0], reverse=True)
    return {"aired": [{"date": e.get("date", ""), "headline": e["headline"], "similarity": round(r, 2)}
                      for r, e in scored[:3] if r >= 0.45]}


class AgentEditor:
    """The LLM editor plus tools and a check-and-repair loop.

    Draft with tools (OpenAI) or one plain call (other providers), then check the picks in
    code: exactly n, no duplicates, not already aired, fresh, links taken from today's
    material. Failing slots go back to the model up to ``max_repairs`` times.
    """

    name = "agent"

    def __init__(self, llm: LLM, max_age_hours: float, max_repairs: int = 2):
        self.llm, self.max_age_hours, self.max_repairs = llm, max_age_hours, max_repairs
        self.base = LLMEditor(llm, max_age_hours)

    def _tools(self, candidates: list[Story], seen: SeenStore) -> list[Tool]:
        query = strict_object({"query": {"type": "string", "description": "a few keywords, e.g. a company and product"}})
        headline = strict_object({"headline": {"type": "string"}})
        return [
            Tool("search_candidates", "Search today's newsletters and feed stories for an event.", query,
                 lambda query: candidate_search(candidates, query)),
            Tool("aired_lookup", "Check whether a similar story aired in the last two weeks.", headline,
                 lambda headline: aired_search(seen, headline)),
        ]

    def pick(self, candidates: list[Story], n: int, seen: SeenStore) -> list[Story]:
        prompt = self.base._prompt(candidates, n, seen)
        if hasattr(self.llm, "run_tools"):
            data = self.llm.run_tools(EDITOR_SYSTEM + AGENT_TOOLS_NOTE, prompt, self._tools(candidates, seen),
                                      stage="editor", schema=EDITOR_SCHEMA, max_turns=4)
        else:
            data = self.llm.json(EDITOR_SYSTEM, prompt, stage="editor", schema=EDITOR_SCHEMA)
        picks = LLMEditor.to_stories(data, candidates, n)
        for round_no in range(1, self.max_repairs + 1):
            issues = fatal(check_picks(picks, candidates, n, seen.urls, seen.recent_headlines(SeenStore.KEEP_DAYS),
                                       self.max_age_hours))
            if not issues:
                break
            log.info("      editor repair round %d: %s", round_no, "; ".join(i.detail for i in issues))
            rows = [{"headline": s.headline, "url": s.url, "key_fact": s.key_fact, "summary": s.summary,
                     "outlets": s.outlets} for s in picks]
            repair = REPAIR_PROMPT.format(picks=json.dumps(rows, indent=1), n=n,
                                          problems="\n".join(i.line() for i in issues))
            try:
                data = self.llm.json(EDITOR_SYSTEM, prompt + repair, stage=f"editor-repair-{round_no}",
                                     schema=EDITOR_SCHEMA)
            except BudgetExceeded as exc:
                log.warning("      %s", exc)
                break
            picks = LLMEditor.to_stories(data, candidates, n) or picks
        return picks


def build_editor(llm: LLM | None, max_age_hours: float, agents: bool = False, max_repairs: int = 2) -> Editor:
    if not llm:
        return HeuristicEditor(max_age_hours)
    return AgentEditor(llm, max_age_hours, max_repairs) if agents else LLMEditor(llm, max_age_hours)


def settle(picks: list[Story], candidates: list[Story], n: int, seen: SeenStore,
           max_age_hours: float) -> list[Story]:
    """Keep picks in order while they pass the checks, up to n.

    A link that isn't in today's material is dropped (the story stays); duplicates, repeats,
    stale and sample stories are dropped.
    """
    kept: list[Story] = []
    aired = seen.recent_headlines(SeenStore.KEEP_DAYS)
    for s in picks:
        if len(kept) == n:
            break
        issues = [i for i in check_picks(kept + [s], candidates, n, seen.urls, aired, max_age_hours)
                  if i.index == len(kept) and i.fatal]
        if any(i.code == "url_not_in_sources" for i in issues):
            log.warning("      dropping made-up link %s for %r", s.url, s.headline or s.title)
            s.url = ""
            issues = [i for i in issues if i.code != "url_not_in_sources"]
        if issues:
            log.warning("      dropping %r: %s", s.headline or s.title, "; ".join(i.detail for i in issues))
            continue
        kept.append(s)
    return kept


def pick_with_fallback(editor: Editor, candidates: list[Story], n: int, seen: SeenStore,
                       max_age_hours: float) -> list[Story]:
    """The editor's picks, checked, then topped up from the keyword ranking if any were dropped."""
    picked: list[Story] = []
    try:
        picked = editor.pick(candidates, n, seen)
    except Exception as exc:
        log.warning("%s editor failed (%s); falling back to heuristic", editor.name, exc)
    pool = HeuristicEditor(max_age_hours).pick(candidates, len(candidates), seen)
    kept = settle(picked, candidates, n, seen, max_age_hours)
    if len(kept) < n:
        if picked and not isinstance(editor, HeuristicEditor):
            log.warning("%s editor gave %d usable stories of %d; topping up from the keyword ranking",
                        editor.name, len(kept), n)
        kept = settle(kept + [s for s in pool if s not in kept], candidates, n, seen, max_age_hours)
    return kept
