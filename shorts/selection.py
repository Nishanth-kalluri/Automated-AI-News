"""Stage 2: the editor picks today's stories.

``LLMEditor`` reads the newsletter issues plus feed headlines, merges coverage of the same
story, and picks the most important ones. ``HeuristicEditor`` is the no-key fallback that
scores feed headlines on AI relevance, freshness and popularity. With web tools
(``SHORTS_WEB=on``) the editor also sees how widely the top headlines are covered, can search
the web for a pick's primary source, and names a few alternates for the researchers' swaps.
"""
from __future__ import annotations

import json
import logging
import math
import re
from datetime import date, datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlparse

from .checks import check_picks, fatal, norm_url, same_event, similar
from .content import description_problem, is_aggregator_url, is_newsletter_url, publisher_name
from .llm import LLM, BudgetExceeded, Tool, capped, strict_object
from .models import Story

if TYPE_CHECKING:
    from .coverage import Coverage
    from .web import Tavily

log = logging.getLogger(__name__)

AI_TERMS = {
    "ai": 1.0, "llm": 1.0, "gpt": 1.0, "claude": 1.0, "gemini": 1.0, "llama": 0.8,
    "openai": 1.0, "anthropic": 1.0, "deepmind": 1.0, "mistral": 0.8, "nvidia": 0.6,
    "model": 0.5, "agent": 0.7, "agents": 0.7, "chatbot": 0.6, "neural": 0.5,
    "machine learning": 0.8, "artificial intelligence": 1.0, "inference": 0.4,
    "benchmark": 0.4, "open-weight": 0.8, "open source": 0.4, "regulation": 0.4,
}
MAX_HEADLINES_FOR_LLM = 80
COVERAGE_IN_PROMPT = 10  # top feed headlines shown with their coverage
MAX_COVERAGE_CALLS = 10
MAX_EDITOR_SEARCHES = 3
MAX_ALTERNATES = 3
# Newsletter click-tracking links run to hundreds of characters, and an issue has dozens of them: in the
# editor's prompt, a link longer than this is "link:N on <site>" and becomes the real link again in a pick.
LONG_LINK = 100
_LINK_RE = re.compile(r"https?://[^\s)>\]\"'<]+")
_LINK_REF = re.compile(r"\s*link:(\d+)\b.*", re.I | re.S)
# Shares of the run's budget: the editor's tool turns end once it has spent the first (its answer comes
# next, without tools), and no repair round starts after the second, so the researchers and the writer
# always have money left.
EDITOR_TOOLS_SHARE = 0.3
EDITOR_REPAIR_SHARE = 0.5


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
                     and s.kind == "article" and not is_aggregator_url(s.url) and not is_newsletter_url(s.url)),
                    key=lambda s: s.score, reverse=True)
    chosen: list[Story] = []
    for s in ranked:
        if any(SequenceMatcher(None, _norm(s.title), _norm(c.title)).ratio() > 0.6 or same_event(s.title, c.title)
               for c in chosen):
            continue  # same story from another outlet
        chosen.append(s)
        if len(chosen) == n:
            break
    return chosen


class ShortLinks:
    """Long links in the editor's prompt as "link:N on <site>", and back to the real link in its picks."""

    def __init__(self):
        self.urls: list[str] = []
        self._ids: dict[str, int] = {}

    def shorten(self, text: str) -> str:
        def short(m: re.Match) -> str:
            url = m.group(0).rstrip(".,;:")
            if len(url) <= LONG_LINK:
                return m.group(0)
            if url not in self._ids:
                self.urls.append(url)
                self._ids[url] = len(self.urls)
            return f"link:{self._ids[url]} on {urlparse(url).hostname or 'a link'}{m.group(0)[len(url):]}"

        return _LINK_RE.sub(short, text or "")

    def expand(self, url: str) -> str:
        """The real link for "link:N" (also "link:N on site"); any other url as it is."""
        m = _LINK_REF.fullmatch(url or "")
        if m and 0 < int(m.group(1)) <= len(self.urls):
            return self.urls[int(m.group(1)) - 1]
        return url


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
- Skip Hacker News and Reddit threads, and anything whose only source is a forum discussion: those are
  reactions to news, not news. Coverage numbers (points, comments, outlet counts) are only for ranking.
- Every pick needs enough facts in the texts for a solid 15 second segment. Fewer, stronger stories beat
  padding: return fewer than requested rather than a thin, vague or duplicate story.
- Use only facts present in the texts. Never invent numbers, names or quotes.
- Summaries describe the event itself. Never mention newsletters or Hacker News, never say what people on
  Reddit or in comments said, never give points, upvotes or comment counts, and never copy a newsletter's or
  site's lines about itself ("in our newsletter", "sign up"). News about Reddit the company is fine.

Return JSON: {"stories": [ ... ]} with up to the requested number of stories, most important first.
Every url must be copied exactly from the texts; never build or guess a link. Long links are shortened
to "link:N on <site>": for one of those, the url is just "link:N".
Each story: {
  "headline": "on-screen headline, at most 8 words",
  "summary": "3 to 5 sentences with the concrete facts from the texts: who, what, numbers, why it matters",
  "key_fact": "the single most striking number or fact, at most 7 words, or empty string",
  "url": "the best link for the story found in the texts, preferring the original announcement or article over newsletter links, never the newsletter's own web page; empty string if none",
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
EDITOR_WEB_SCHEMA = strict_object({"stories": {"type": "array", "items": PICK_SCHEMA},
                                   "alternates": {"type": "array", "items": PICK_SCHEMA}})


def _coverage_note(row: dict | None) -> str:
    if not row or (row.get("hn_points") is None and row.get("news_outlets") is None):
        return " | coverage unknown"
    from .coverage import Coverage

    return f" | coverage: {Coverage.line(row)}"


class LLMEditor:
    name = "llm"

    def __init__(self, llm: LLM, max_age_hours: float, coverage: Coverage | None = None):
        self.llm, self.max_age_hours, self.coverage = llm, max_age_hours, coverage
        self.links = ShortLinks()  # the last prompt's shortened links

    def _prompt(self, candidates: list[Story], n: int, seen: SeenStore) -> str:
        self.links = ShortLinks()
        newsletters = [s for s in candidates if s.kind == "newsletter"]
        now = datetime.now(timezone.utc)
        articles = sorted((s for s in candidates if s.kind == "article" and s.title),
                          key=lambda s: score(s, now, self.max_age_hours), reverse=True)
        articles = [s for s in articles if s.url not in seen.urls][:MAX_HEADLINES_FOR_LLM]
        parts = [f"Today is {date.today().isoformat()}. Pick up to {n} stories: fewer if there aren't {n} solid, "
                 "different news events.\n"]
        for i, s in enumerate(newsletters, 1):
            parts.append(f"=== NEWSLETTER {i}: {s.source} | {s.title} | {s.published:%Y-%m-%d %H:%M} UTC ===\n"
                         f"{self.links.shorten(s.body)}\n")
        if articles:
            notes = self._coverage_notes(articles)
            parts.append("=== FEED HEADLINES (source | published | title | url | snippet"
                         f"{' | coverage' if notes else ''}) ===")
            parts += [f"- {s.source} | {s.published:%m-%d %H:%M} | {s.title} | {self.links.shorten(s.url)} | "
                      f"{s.summary[:200]}"
                      f"{notes.get(id(s), '')}" for s in articles]
        recent = seen.recent_headlines()
        if recent:
            parts.append("\n=== ALREADY COVERED IN RECENT EPISODES ===")
            parts += [f"- {h}" for h in recent]
        return "\n".join(parts)

    def _coverage_notes(self, articles: list[Story]) -> dict[int, str]:
        """Coverage for the top feed headlines (one per event), keyed by id(story); {} without coverage."""
        if self.coverage is None:
            return {}
        top: list[Story] = []
        for s in articles:
            if not any(same_event(s.title, t.title) for t in top):
                top.append(s)
            if len(top) == COVERAGE_IN_PROMPT:
                break
        try:
            rows = self.coverage.lookup_many([s.title for s in top], timeout=30)
        except Exception as exc:  # the editor still has the newsletters and headlines
            log.warning("      coverage lookup failed (%s)", exc)
            rows = {}
        return {id(s): _coverage_note(rows.get(s.title)) for s in top}

    def pick(self, candidates: list[Story], n: int, seen: SeenStore) -> list[Story]:
        system = EDITOR_SYSTEM + (COVERAGE_NOTE if self.coverage is not None else "")
        data = self.llm.json(system, self._prompt(candidates, n, seen), stage="editor", schema=EDITOR_SCHEMA)
        return self.to_stories(data, candidates, n, links=self.links)

    @staticmethod
    def to_stories(data: dict, candidates: list[Story], n: int, quiet: bool = False,
                   links: ShortLinks | None = None) -> list[Story]:
        """Editor JSON -> Stories. A link that matches a feed article or a web search hit takes its
        date and publisher. ``links`` turns the prompt's "link:N" back into real links."""
        now = datetime.now(timezone.utc)
        articles = {norm_url(c.url): c for c in candidates if c.kind in ("article", "web") and c.url}
        picked = []
        for item in (data.get("stories") or [])[:n]:
            outlets = [o for o in item.get("outlets", []) if o] or ["AI newsletters"]
            url = (item.get("url") or "").strip()
            url = links.expand(url) if links else url
            if is_newsletter_url(url):
                url = ""  # the newsletter's own web copy is never the source; research finds the original
            match = articles.get(norm_url(url))
            if match and match.source and match.source not in outlets:
                outlets.append(match.source)
            picked.append(Story(
                title=item.get("headline", "").strip(),
                url=url,
                # the publisher to credit on screen: never a newsletter or a forum
                source=(match.source if match else "") or publisher_name(url),
                published=match.published if match else now,
                summary=item.get("summary", "").strip(),
                popularity=match.popularity if match else 0.0,
                headline=item.get("headline", "").strip(),
                key_fact=item.get("key_fact", "").strip(),
                outlets=outlets,
            ))
        picked = [s for s in picked if s.headline and s.summary]
        if len(picked) < n and not quiet:
            log.warning("editor returned %d of %d stories", len(picked), n)
        return picked


AGENT_TOOLS_NOTE = """

You have two tools. Use them before you answer; a few calls are enough.
- search_candidates(query): searches today's newsletters and feeds. Use it to find every outlet that covered an
  event (more outlets means a bigger story) and the best link for it. A link must come from today's material.
- aired_lookup(headline): checks whether a story already aired in the last two weeks."""

COVERAGE_NOTE = """

The top feed headlines show how widely each event is covered right now: Hacker News points and how many
outlets Google News lists in the last 2 days. Unknown means unknown, not low. Use it to rank stories of
similar importance and to spot the day's big stories; never drop an important launch or paper because its
coverage is low or unknown."""

COVERAGE_TOOL_NOTE = """
- coverage(headline): the same coverage numbers for any event (null means unknown), at most 10 calls."""

SEARCH_TOOL_NOTE = """
- web_search(query), at most 3 calls: only to find the primary source (the company's post, the paper) of a
  pick without a good link. Its result links count as today's material, and a link's date becomes the
  story's date, so never use a source published before the stories you are covering."""

ALTERNATES_NOTE = """

Also return "alternates": up to 3 next-best stories in the same format, each a different event from your
picks. They are used only if a researcher finds that a pick is wrong, stale or thin."""

REPAIR_PROMPT = """

=== YOUR PICKS SO FAR ===
{picks}

=== PROBLEMS TO FIX ===
{problems}

Return the full list again, up to {n} stories. Keep the good ones exactly as they are and replace or fix only
the ones with problems. A replacement must be a different real event from the texts above; if there is no good
replacement, leave the story out."""


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
    return {"aired": [{"date": e.get("date", ""), "headline": e["headline"], "similarity": round(r, 2),
                       "same_event": same_event(headline, e["headline"])}
                      for r, e in scored[:3] if r >= 0.3]}


class AgentEditor:
    """The LLM editor plus tools and a check-and-repair loop.

    Draft with tools (OpenAI) or one plain call (other providers), then check the picks in
    code: exactly n, no duplicates, not already aired, fresh, links taken from today's
    material. Failing slots go back to the model up to ``max_repairs`` times.
    """

    name = "agent"

    def __init__(self, llm: LLM, max_age_hours: float, max_repairs: int = 2, *,
                 coverage: Coverage | None = None, tavily: Tavily | None = None, min_stories: int | None = None):
        self.llm, self.max_age_hours, self.max_repairs = llm, max_age_hours, max_repairs
        self.min_stories = min_stories  # fewer picks than this goes back for repair; None means n
        self.coverage, self.tavily = coverage, tavily
        self.web = coverage is not None or tavily is not None
        self.base = LLMEditor(llm, max_age_hours, coverage)
        self.alternates: list[Story] = []  # next-best picks, for the researchers' swaps

    def _tools(self, candidates: list[Story], seen: SeenStore) -> list[Tool]:
        query = strict_object({"query": {"type": "string", "description": "a few keywords, e.g. a company and product"}})
        headline = strict_object({"headline": {"type": "string"}})
        tools = [
            Tool("search_candidates", "Search today's newsletters and feed stories for an event.", query,
                 lambda query: candidate_search(candidates, query)),
            Tool("aired_lookup", "Check whether a similar story aired in the last two weeks.", headline,
                 lambda headline: aired_search(seen, headline)),
        ]
        if self.coverage is not None:
            tools += capped([Tool("coverage", "How widely an event is covered right now (null means unknown).",
                                  headline, lambda headline: self.coverage.lookup(headline))], MAX_COVERAGE_CALLS)
        if self.tavily is not None and self.tavily.usable:
            tools += capped([Tool("web_search", "Search the web (past week) for a pick's primary source or "
                                  "first-report date.", query, lambda query: self._search(candidates, query))],
                            MAX_EDITOR_SEARCHES)
        return tools

    def _search(self, candidates: list[Story], query: str) -> str:
        """A Tavily search whose hits join today's material, so a primary link found here passes the checks."""
        from .web import WebUnavailable

        try:
            docs = self.tavily.search(query, 5)
        except WebUnavailable as exc:
            return f"error: {exc}"
        now = datetime.now(timezone.utc)
        known = {norm_url(c.url) for c in candidates if c.url}
        rows = []
        for i, d in enumerate(docs, 1):
            if norm_url(d.url) not in known:
                known.add(norm_url(d.url))
                candidates.append(Story(title=d.title or d.url, url=d.url, source=publisher_name(d.url),
                                        published=_published(d.published, now), summary=d.snippet, kind="web"))
            rows.append(f"{i}. {d.title} | {d.url} | {d.published or 'date unknown'}\n   {d.snippet}")
        return "\n".join(rows) or "no results"

    def pick(self, candidates: list[Story], n: int, seen: SeenStore) -> list[Story]:
        prompt = self.base._prompt(candidates, n, seen)
        schema = EDITOR_WEB_SCHEMA if self.web else EDITOR_SCHEMA
        coverage_note = COVERAGE_NOTE if self.coverage is not None else ""
        alternates_note = ALTERNATES_NOTE if self.web else ""
        if hasattr(self.llm, "run_tools"):
            tools = self._tools(candidates, seen)
            names = {t.name for t in tools}
            tools_note = AGENT_TOOLS_NOTE if len(tools) == 2 else AGENT_TOOLS_NOTE.replace("two tools", "these tools")
            system = (EDITOR_SYSTEM + coverage_note + tools_note
                      + (COVERAGE_TOOL_NOTE if "coverage" in names else "")
                      + (SEARCH_TOOL_NOTE if "web_search" in names else "") + alternates_note)
            data = self.llm.run_tools(system, prompt, tools, stage="editor", schema=schema, max_turns=4,
                                      spend_limit=self._share(EDITOR_TOOLS_SHARE))
        else:
            data = self.llm.json(EDITOR_SYSTEM + coverage_note + alternates_note, prompt, stage="editor",
                                 schema=schema)
        links = self.base.links
        picks = LLMEditor.to_stories(data, candidates, n, links=links)
        if self.web:
            self.alternates = LLMEditor.to_stories({"stories": data.get("alternates") or []}, candidates,
                                                   MAX_ALTERNATES, quiet=True, links=links)
        for round_no in range(1, self.max_repairs + 1):
            issues = fatal(check_picks(picks, candidates, n, seen.urls, seen.recent_headlines(SeenStore.KEEP_DAYS),
                                       self.max_age_hours, min_n=self.min_stories))
            if not issues:
                break
            log.info("      editor repair round %d: %s", round_no, "; ".join(i.detail for i in issues))
            spent, limit = self._spent(), self._share(EDITOR_REPAIR_SHARE)
            if limit is not None and spent >= limit:
                log.warning("      no repair: the editor has spent $%.2f, its share of the run's budget", spent)
                break
            rows = [{"headline": s.headline, "url": s.url, "key_fact": s.key_fact, "summary": s.summary,
                     "outlets": s.outlets} for s in picks]
            repair = REPAIR_PROMPT.format(picks=json.dumps(rows, indent=1), n=n,
                                          problems="\n".join(i.line() for i in issues))
            try:
                data = self.llm.json(EDITOR_SYSTEM + coverage_note, prompt + repair,
                                     stage=f"editor-repair-{round_no}", schema=EDITOR_SCHEMA)
                picks = LLMEditor.to_stories(data, candidates, n, links=links) or picks
            except BudgetExceeded as exc:
                log.warning("      %s", exc)
                break
            except Exception as exc:  # keep the draft; settle() and the top-up take it from here
                log.warning("      editor repair failed (%s); keeping the current picks", exc)
                break
        return picks


    def _share(self, share: float) -> float | None:
        """That share of the run's budget in dollars; None without a cap."""
        cap = getattr(getattr(self.llm, "usage", None), "run_cap_usd", None)
        return cap * share if isinstance(cap, (int, float)) and math.isfinite(cap) else None

    def _spent(self) -> float:
        usage = getattr(self.llm, "usage", None)
        return usage.stage_usd("editor") if usage is not None else 0.0


def _published(day: str, now: datetime) -> datetime:
    """A search hit's date (YYYY-MM-DD) as the end of that day, capped at now; now if unknown."""
    try:
        end = datetime.fromisoformat(day[:10]).replace(hour=23, minute=59, tzinfo=timezone.utc)
    except ValueError:
        return now
    return min(end, now)


def build_editor(llm: LLM | None, max_age_hours: float, agents: bool = False, max_repairs: int = 2, *,
                 coverage: Coverage | None = None, tavily: Tavily | None = None,
                 min_stories: int | None = None) -> Editor:
    if not llm:
        return HeuristicEditor(max_age_hours)
    if agents:
        return AgentEditor(llm, max_age_hours, max_repairs, coverage=coverage, tavily=tavily,
                           min_stories=min_stories)
    return LLMEditor(llm, max_age_hours, coverage)


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
            if s.source == publisher_name(s.url):
                s.source = ""  # the credit came from the made-up link
            s.url = ""
            issues = [i for i in issues if i.code != "url_not_in_sources"]
        if issues:
            log.warning("      dropping %r: %s", s.headline or s.title, "; ".join(i.detail for i in issues))
            continue
        kept.append(s)
    return kept


def pick_with_fallback(editor: Editor, candidates: list[Story], n: int, seen: SeenStore,
                       max_age_hours: float, min_n: int | None = None) -> list[Story]:
    """The editor's picks, checked. Fewer than ``n`` is fine; below ``min_n`` (default ``n``) they are
    topped up from the keyword ranking, but only with stories whose feed text really describes them."""
    min_n = n if min_n is None else min(min_n, n)
    picked: list[Story] = []
    try:
        picked = editor.pick(candidates, n, seen)
    except Exception as exc:
        log.warning("%s editor failed (%s); falling back to heuristic", editor.name, exc)
    pool = [s for s in HeuristicEditor(max_age_hours).pick(candidates, len(candidates), seen)
            if not description_problem(s)]
    kept = settle(picked, candidates, n, seen, max_age_hours)
    if len(kept) < min_n:
        if picked and not isinstance(editor, HeuristicEditor):
            log.warning("%s editor gave %d usable stories (at least %d needed); topping up from the keyword "
                        "ranking", editor.name, len(kept), min_n)
        kept = settle(kept + [s for s in pool if s not in kept], candidates, n if not picked else min_n, seen,
                      max_age_hours)
    return kept


DEDUPE_SYSTEM = """You check the running order of a daily AI news show for duplicates: two stories about the
same real-world event (the same launch, deal, paper, lawsuit, incident or announcement), even when the
headlines are worded differently or come from different outlets. Related but separate events are not
duplicates. Return JSON {"duplicates": [{"story": <the later story's number>, "same_as": <the earlier one's>}]},
an empty list when every story is a different event."""
DEDUPE_SCHEMA = strict_object({"duplicates": {"type": "array", "items": strict_object({
    "story": {"type": "integer"}, "same_as": {"type": "integer"}})}})


def drop_duplicates(llm: LLM | None, stories: list[Story]) -> list[Story]:
    """A model's second look for picks that are the same event; the later one goes. Code checks
    catch most duplicates first; this catches differently worded ones. Never fails the run."""
    if llm is None or len(stories) < 2:
        return stories
    rows = "\n".join(f"{i}. {s.headline or s.title}: {s.summary[:300]}" for i, s in enumerate(stories, 1))
    try:
        data = llm.json(DEDUPE_SYSTEM, rows, stage="dedupe", schema=DEDUPE_SCHEMA)
    except Exception as exc:
        log.warning("      duplicate check failed (%s); keeping the picks", exc)
        return stories
    drop = set()
    for d in data.get("duplicates") or []:
        try:
            later, earlier = int(d.get("story")), int(d.get("same_as"))
        except (AttributeError, TypeError, ValueError):
            continue
        if 0 < earlier < later <= len(stories):
            drop.add(later - 1)
    for i in sorted(drop):
        log.warning("      dropping %r: same event as an earlier story", stories[i].headline or stories[i].title)
    return [s for i, s in enumerate(stories) if i not in drop]
