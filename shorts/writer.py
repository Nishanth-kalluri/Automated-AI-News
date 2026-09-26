"""Stage 4: turn the picked stories into an episode: intro, one segment per story, outro.

``LLMWriter`` makes one call. ``CriticWriter`` (the agent) drafts with it, checks the script in
code and with a fact-checking critic, and sends only the failing segments back for rewriting.
Whatever still fails after the repair rounds is replaced segment by segment with template
text, so one bad segment never costs the whole script.
"""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Protocol

from .checks import (STORY_WORDS, TARGET_MAX_SECONDS, WORDS_PER_SECOND, Issue, episode_material, fatal,
                     lint_episode, predicted_seconds, unsupported_numbers)
from .llm import LLM, BudgetExceeded, strict_object
from .models import Episode, Segment, Story

log = logging.getLogger(__name__)
PERSONA_FILE = Path(__file__).with_name("persona.md")
SIGN_OFF = "That's the news from the pond. See you tomorrow!"
# Problems about the intro or outro only; they don't open the story segments for rewriting.
FRAME_CODES = ("missing_intro", "missing_outro", "intro_problem", "outro_problem")

WRITER_SYSTEM = """You write the script for a daily 2 minute vertical YouTube Short.

{persona}

Format:
- "intro": 12 to 18 words. A hook that teases the biggest story, then says it's {n} AI stories today.
- "segments": exactly one per story, in the given order. Each is {lo} to {hi} words (about 15 seconds spoken):
  what happened with the key specifics, then one short line on why it matters.
- "outro": 10 to 16 words ending with the sign-off.
- Written to be heard: short sentences, no parentheses, no URLs, no emoji, spell out symbols ("percent", "dollars").
- Only use facts given in the story material. Never invent numbers, names or quotes.

Return JSON:
{{"title": "YouTube title under 70 characters, no hashtags",
  "description": "2 to 3 sentences describing today's episode",
  "tags": ["5 to 10 tags"],
  "intro": "...",
  "segments": [{{"headline": "on-screen headline, max 8 words", "key_fact": "max 7 words or empty", "text": "..."}}],
  "outro": "..."}}"""

SEGMENT_SCHEMA = strict_object({"headline": {"type": "string"}, "key_fact": {"type": "string"},
                                "text": {"type": "string"}})
SCRIPT_SCHEMA = strict_object({
    "title": {"type": "string"},
    "description": {"type": "string"},
    "tags": {"type": "array", "items": {"type": "string"}},
    "intro": {"type": "string"},
    "segments": {"type": "array", "items": SEGMENT_SCHEMA},
    "outro": {"type": "string"},
})
REVISION_SCHEMA = strict_object({
    "intro": {"type": "string"},
    "outro": {"type": "string"},
    "segments": {"type": "array", "items": strict_object({
        "story": {"type": "integer", "description": "story number, starting at 1"},
        "headline": {"type": "string"}, "key_fact": {"type": "string"}, "text": {"type": "string"}})},
})
REVIEW_SCHEMA = strict_object({
    "intro": {"type": "array", "items": {"type": "string"},
              "description": "claims in the intro that the material does not support"},
    "outro": {"type": "array", "items": {"type": "string"},
              "description": "claims in the outro that the material does not support"},
    "segments": {"type": "array", "items": strict_object({
        "story": {"type": "integer", "description": "story number, starting at 1"},
        "unsupported": {"type": "array", "items": {"type": "string"},
                        "description": "claims in the segment or its key fact that the material does not support"}})},
})

CRITIC_SYSTEM = """You fact-check the script of a daily AI news Short against the source material for each story.
For every story segment, list each claim (a name, number, date, quote or event) in the segment text or its key
fact that the material does not support. Check the intro and outro the same way against all the stories.
Paraphrase and rounding are fine; new facts, wrong numbers and claims about the wrong company are not.
Return an empty list for a segment with no problems."""

REVISE_PROMPT = """Here is today's script and the story material. Fix only the problems listed below.

=== PROBLEMS ===
{problems}

=== CURRENT SCRIPT ===
{script}

=== STORY MATERIAL ===
{material}

Return JSON with "intro", "outro" (unchanged unless a problem is about them) and "segments" holding only
the segments you changed, each with its story number. Keep every rule from your instructions."""


def load_persona(show: str, host: str) -> str:
    return PERSONA_FILE.read_text().format(show=show, host=host)


class ScriptWriter(Protocol):
    name: str

    def write(self, stories: list[Story]) -> Episode: ...


def _story_block(stories: list[Story]) -> str:
    out = []
    for i, s in enumerate(stories, 1):
        part = f"STORY {i}: {s.headline or s.title}\nCovered by: {', '.join(s.outlets or [s.source])}\n"
        part += f"Key fact: {s.key_fact}\nSummary: {s.summary}\n"
        if s.body:
            part += f"Article text:\n{s.body}\n"
        out.append(part)
    return "\n".join(out)


def description_footer(stories: list[Story]) -> str:
    lines = [f"- {s.headline or s.title}: {s.url}" if s.url else f"- {s.headline or s.title} ({s.source})"
             for s in stories]
    return "\n\nSources:\n" + "\n".join(lines) + "\n\nMade with AI. #AI #AINews #Shorts"


def _story_segment(story: Story, text: str, headline: str = "", key_fact: str = "") -> Segment:
    return Segment(kind="story", text=text.strip(), headline=(headline or story.headline or story.title).strip(),
                   key_fact=(key_fact or story.key_fact).strip(), source=story.source, url=story.url)


def _first_words(text: str, limit: int) -> str:
    """At most ``limit`` words, cut at a sentence end when one is reasonably close."""
    clipped = " ".join(text.split()[:limit])
    if clipped.endswith((".", "!", "?")):
        return clipped
    end = max(clipped.rfind(". "), clipped.rfind("! "), clipped.rfind("? "))
    if end > len(clipped) // 2:
        return clipped[:end + 1]
    return clipped.rstrip(",;:") + "."


def template_segment(story: Story) -> Segment:
    body = _first_words(story.summary, STORY_WORDS[1] - len(story.title.split())) if story.summary else ""
    return _story_segment(story, f"{(story.headline or story.title).rstrip('.')}. {body}".strip())


def assemble(data: dict, stories: list[Story], *, strict: bool = True) -> Episode:
    """Writer JSON -> Episode. With ``strict=False`` a missing segment gets template text instead of failing."""
    items = data.get("segments", [])
    if strict and len(items) != len(stories):
        raise ValueError(f"writer returned {len(items)} segments for {len(stories)} stories")
    if len(items) != len(stories):
        log.warning("writer returned %d segments for %d stories; filling the gaps from templates",
                    len(items), len(stories))
    segments = [Segment(kind="intro", text=(data.get("intro") or "").strip())]
    for i, story in enumerate(stories):
        item = items[i] if i < len(items) else None
        if item and (item.get("text") or "").strip():
            segments.append(_story_segment(story, item["text"], item.get("headline", ""), item.get("key_fact", "")))
        else:
            segments.append(template_segment(story))
    segments.append(Segment(kind="outro", text=(data.get("outro") or "").strip()))
    if strict and not (segments[0].text and segments[-1].text):
        raise ValueError("writer left out the intro or outro")
    return Episode(
        title=(data.get("title") or "").strip()[:95],
        description=(data.get("description") or "").strip() + description_footer(stories),
        tags=[t for t in data.get("tags", []) if isinstance(t, str)][:15],
        segments=segments,
        stories=stories,
    )


class LLMWriter:
    name = "llm"

    def __init__(self, llm: LLM, show: str, host: str):
        self.llm, self.show, self.host = llm, show, host

    def system(self, n: int) -> str:
        return WRITER_SYSTEM.format(persona=load_persona(self.show, self.host), n=n,
                                    lo=STORY_WORDS[0], hi=STORY_WORDS[1])

    def draft(self, stories: list[Story]) -> dict:
        user = f"Today is {date.today():%A, %B %d, %Y}.\n\n{_story_block(stories)}"
        return self.llm.json(self.system(len(stories)), user, stage="writer", schema=SCRIPT_SCHEMA)

    def write(self, stories: list[Story]) -> Episode:
        episode = assemble(self.draft(stories), stories)
        for seg in episode.story_segments:
            words = len(seg.text.split())
            if not STORY_WORDS[0] - 8 <= words <= STORY_WORDS[1] + 10:
                log.warning("segment %r is %d words (target %d-%d)", seg.headline, words, *STORY_WORDS)
        return episode


class TemplateWriter:
    """No-key fallback: headline plus the first sentences of each summary."""

    name = "template"

    def __init__(self, show: str, host: str):
        self.show, self.host = show, host

    def intro(self, n: int) -> Segment:
        return Segment(kind="intro", text=f"Quack quack, it's {self.host} on {self.show}! "
                                          f"Here are {n} AI stories you need today.")

    def outro(self) -> Segment:
        return Segment(kind="outro", text=f"Follow for tomorrow's AI news. {SIGN_OFF}")

    def write(self, stories: list[Story]) -> Episode:
        segments = [self.intro(len(stories)), *(template_segment(s) for s in stories), self.outro()]
        return Episode(
            title=f"AI News Today: {stories[0].headline or stories[0].title}"[:95],
            description=f"Today's top {len(stories)} AI stories in two minutes." + description_footer(stories),
            tags=["AI", "AI news", "artificial intelligence", "tech news", "shorts"],
            segments=segments,
            stories=stories,
        )


def _script_rows(episode: Episode) -> str:
    rows = {"intro": episode.segments[0].text, "outro": episode.segments[-1].text,
            "segments": [{"story": i, "headline": s.headline, "key_fact": s.key_fact, "text": s.text,
                          "words": len(s.text.split())}
                         for i, s in enumerate(episode.story_segments, 1)]}
    return json.dumps(rows, indent=1)


class CriticWriter:
    """Writer agent: draft, check in code and with a critic, rewrite only what fails."""

    name = "agent"

    def __init__(self, llm: LLM, critic: LLM, show: str, host: str, max_repairs: int = 2):
        self.base = LLMWriter(llm, show, host)
        self.llm, self.critic, self.max_repairs = llm, critic, max_repairs
        self.template = TemplateWriter(show, host)

    def write(self, stories: list[Story]) -> Episode:
        episode = assemble(self.base.draft(stories), stories, strict=False)
        issues: list[Issue] = []
        for round_no in range(self.max_repairs + 1):
            issues = lint_episode(episode, stories, self.frame())
            try:
                issues += self.review(episode, stories)
            except BudgetExceeded as exc:
                log.warning("      %s", exc)
            except Exception as exc:  # the critic is an extra check, never a reason to fail
                log.warning("      critic failed (%s); using the rule checks only", exc)
            todo = fatal(issues)
            if not todo or round_no == self.max_repairs:
                break
            log.info("      writer repair round %d: %s", round_no + 1, "; ".join(i.line()[2:] for i in todo))
            try:
                episode = self.revise(episode, stories, todo, stage=f"writer-repair-{round_no + 1}")
            except BudgetExceeded as exc:
                log.warning("      %s", exc)
                break
            except Exception as exc:
                log.warning("      rewrite failed (%s)", exc)
                break
        return self.settle(episode, stories, fatal(lint_episode(episode, stories, self.frame())) + [
            i for i in fatal(issues) if i.code in ("unsupported_claim", "intro_problem", "outro_problem")])

    def frame(self) -> str:
        """What the intro and outro may mention besides the stories."""
        return f"{self.base.show} {self.base.host} {date.today():%A, %B %d, %Y}"

    def review(self, episode: Episode, stories: list[Story]) -> list[Issue]:
        user = (f"The intro and outro may name the show ({self.base.show}), the host ({self.base.host}) and today's "
                f"date ({date.today():%A, %B %d, %Y}); those are not claims to check.\n\n"
                f"=== SCRIPT ===\n{_script_rows(episode)}\n\n=== STORY MATERIAL ===\n{_story_block(stories)}")
        data = self.critic.json(CRITIC_SYSTEM, user, stage="critic", schema=REVIEW_SCHEMA)
        issues = []
        for code, key in (("intro_problem", "intro"), ("outro_problem", "outro")):
            claims = [c for c in data.get(key, []) if c]
            if claims:
                issues.append(Issue(code, f"the {key} makes claims the material does not support: "
                                          + "; ".join(claims)))
        for item in data.get("segments", []):
            idx, claims = item.get("story", 0) - 1, [c for c in item.get("unsupported", []) if c]
            if 0 <= idx < len(stories) and claims:
                issues.append(Issue("unsupported_claim", "not supported by the material: " + "; ".join(claims), idx))
        return issues

    def revise(self, episode: Episode, stories: list[Story], problems: list[Issue], *, stage: str) -> Episode:
        user = REVISE_PROMPT.format(problems="\n".join(p.line() for p in problems), script=_script_rows(episode),
                                    material=_story_block(stories))
        data = self.llm.json(self.base.system(len(stories)), user, stage=stage, schema=REVISION_SCHEMA)
        whole_script = any(p.index is None and p.code not in FRAME_CODES for p in problems)
        allowed = set(range(len(stories))) if whole_script else {p.index for p in problems}
        segments = list(episode.segments)
        for item in data.get("segments", []):
            idx = item.get("story", 0) - 1
            if idx in allowed and 0 <= idx < len(stories) and (item.get("text") or "").strip():
                segments[idx + 1] = _story_segment(stories[idx], item["text"], item.get("headline", ""),
                                                   item.get("key_fact", ""))
        if (data.get("intro") or "").strip():
            segments[0] = Segment(kind="intro", text=data["intro"].strip())
        if (data.get("outro") or "").strip():
            segments[-1] = Segment(kind="outro", text=data["outro"].strip())
        return replace(episode, segments=segments)

    def settle(self, episode: Episode, stories: list[Story], remaining: list[Issue]) -> Episode:
        """Last resort for problems the rewrites didn't fix: template text or a trim, segment by segment."""
        segments = list(episode.segments)
        for issue in remaining:
            if issue.code in ("missing_intro", "intro_problem"):
                log.warning("      the intro still has a problem (%s); using the standard intro", issue.detail)
                segments[0] = self.template.intro(len(stories))
            elif issue.code in ("missing_outro", "outro_problem"):
                log.warning("      the outro still has a problem (%s); using the standard outro", issue.detail)
                segments[-1] = self.template.outro()
            elif issue.index is None:
                continue
            elif issue.code == "too_long":
                seg = segments[issue.index + 1]
                segments[issue.index + 1] = replace(seg, text=_first_words(seg.text, STORY_WORDS[1]))
            else:
                log.warning("      story %d still has a problem (%s); using its summary instead",
                            issue.index + 1, issue.detail)
                segments[issue.index + 1] = template_segment(stories[issue.index])
        episode = replace(episode, segments=segments)
        if not episode.title or unsupported_numbers(episode.title, episode_material(stories, self.frame())):
            episode.title = self.template.write(stories).title
        return trim_to_fit(episode, TARGET_MAX_SECONDS)

    def shorten(self, episode: Episode, stories: list[Story], seconds_over: float) -> Episode:
        """A shorter rewrite, checked like the draft: any rewritten part that now fails keeps its old text."""
        words = int(seconds_over * WORDS_PER_SECOND) + 4
        problem = Issue("too_long_total", f"the voiced episode runs {seconds_over:.0f}s too long; cut about {words} "
                                          "words, mostly from the longest segments")
        try:
            shorter = self.revise(episode, stories, [problem], stage="writer-trim")
        except Exception as exc:
            log.warning("      trim rewrite failed (%s); trimming in code", exc)
            return episode
        old = episode.segments
        # A trim changes only the stories' spoken text; the intro, outro, headlines and key facts
        # stay as checked.
        segments = [replace(new, headline=prev.headline, key_fact=prev.key_fact) if new.kind == "story" else prev
                    for new, prev in zip(shorter.segments, old)]
        shorter = replace(shorter, segments=segments)
        issues = fatal(lint_episode(shorter, stories, self.frame()))
        try:
            issues += self.review(shorter, stories)
        except Exception as exc:
            log.warning("      critic failed on the trim (%s); using the rule checks only", exc)
        for issue in issues:
            pos = _segment_position(issue)
            if pos is not None and segments[pos] != old[pos]:
                log.warning("      the trim broke story %d (%s); keeping the old text", pos, issue.detail)
                segments[pos] = old[pos]
        shorter = replace(shorter, segments=segments)
        return shorter if predicted_seconds(shorter) < predicted_seconds(episode) else episode


def _segment_position(issue: Issue) -> int | None:
    """Where in ``episode.segments`` an issue points: 0 intro, -1 outro, i + 1 story i, None the whole episode."""
    if issue.code in ("missing_intro", "intro_problem"):
        return 0
    if issue.code in ("missing_outro", "outro_problem"):
        return -1
    return None if issue.index is None else issue.index + 1


def trim_to_fit(episode: Episode, max_seconds: float) -> Episode:
    """Cut the longest story segments at sentence ends until the predicted length fits."""
    segments = list(episode.segments)
    for _ in range(40):
        ep = replace(episode, segments=segments)
        if predicted_seconds(ep) <= max_seconds:
            return ep
        idx = max(range(1, len(segments) - 1), key=lambda i: len(segments[i].text.split()), default=None)
        if idx is None:
            break
        words = len(segments[idx].text.split())
        if words <= STORY_WORDS[0] - 8:
            break
        segments[idx] = replace(segments[idx], text=_first_words(segments[idx].text, words - 4))
    return replace(episode, segments=segments)


def shorten(writer: ScriptWriter, episode: Episode, stories: list[Story], measured_seconds: float,
            max_seconds: float = TARGET_MAX_SECONDS) -> Episode:
    """The voiced episode came out too long: let the writer agent cut it, then make sure in code."""
    over = measured_seconds - max_seconds
    # The voice runs at its own pace, so scale the target by how far off the prediction was
    # for the script that was actually voiced.
    pace = measured_seconds / max(predicted_seconds(episode), 1.0)
    if isinstance(writer, CriticWriter):
        episode = writer.shorten(episode, stories, over)
    return trim_to_fit(episode, max_seconds / max(pace, 1.0) - 2)


def build_writer(llm: LLM | None, show: str, host: str, *, agents: bool = False, critic: LLM | None = None,
                 max_repairs: int = 2) -> ScriptWriter:
    if not llm:
        return TemplateWriter(show, host)
    return CriticWriter(llm, critic or llm, show, host, max_repairs) if agents else LLMWriter(llm, show, host)


def write_episode(writer: ScriptWriter, stories: list[Story], show: str, host: str) -> Episode:
    try:
        return writer.write(stories)
    except Exception as exc:
        if isinstance(writer, TemplateWriter):
            raise
        log.warning("%s writer failed (%s); falling back to template", writer.name, exc)
        return TemplateWriter(show, host).write(stories)


def episode_json(episode: Episode) -> str:
    from dataclasses import asdict

    return json.dumps(asdict(episode), indent=2, default=str)
