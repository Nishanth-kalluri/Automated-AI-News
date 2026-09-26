"""Stage 4: turn the picked stories into an episode: intro, one segment per story, outro."""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Protocol

from .llm import LLM
from .models import Episode, Segment, Story

log = logging.getLogger(__name__)
PERSONA_FILE = Path(__file__).with_name("persona.md")
STORY_WORDS = (30, 42)  # ~15 seconds of speech each
SIGN_OFF = "That's the news from the pond. See you tomorrow!"

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


class LLMWriter:
    name = "llm"

    def __init__(self, llm: LLM, show: str, host: str):
        self.llm, self.show, self.host = llm, show, host

    def write(self, stories: list[Story]) -> Episode:
        system = WRITER_SYSTEM.format(persona=load_persona(self.show, self.host), n=len(stories),
                                      lo=STORY_WORDS[0], hi=STORY_WORDS[1])
        user = f"Today is {date.today():%A, %B %d, %Y}.\n\n{_story_block(stories)}"
        data = self.llm.json(system, user, stage="writer")
        items = data.get("segments", [])
        if len(items) != len(stories):
            raise ValueError(f"writer returned {len(items)} segments for {len(stories)} stories")
        segments = [Segment(kind="intro", text=data["intro"].strip())]
        for story, item in zip(stories, items):
            seg = _story_segment(story, item["text"], item.get("headline", ""), item.get("key_fact", ""))
            words = len(seg.text.split())
            if not STORY_WORDS[0] - 8 <= words <= STORY_WORDS[1] + 10:
                log.warning("segment %r is %d words (target %d-%d)", seg.headline, words, *STORY_WORDS)
            segments.append(seg)
        segments.append(Segment(kind="outro", text=data["outro"].strip()))
        return Episode(
            title=data["title"].strip()[:95],
            description=data.get("description", "").strip() + description_footer(stories),
            tags=[t for t in data.get("tags", []) if isinstance(t, str)][:15],
            segments=segments,
            stories=stories,
        )


def _first_words(text: str, limit: int) -> str:
    """At most ``limit`` words, cut at a sentence end when one is reasonably close."""
    clipped = " ".join(text.split()[:limit])
    if clipped.endswith((".", "!", "?")):
        return clipped
    end = max(clipped.rfind(". "), clipped.rfind("! "), clipped.rfind("? "))
    if end > len(clipped) // 2:
        return clipped[:end + 1]
    return clipped.rstrip(",;:") + "."


class TemplateWriter:
    """No-key fallback: headline plus the first sentences of each summary."""

    name = "template"

    def __init__(self, show: str, host: str):
        self.show, self.host = show, host

    def write(self, stories: list[Story]) -> Episode:
        segments = [Segment(kind="intro", text=f"Quack quack, it's {self.host} on {self.show}! "
                                               f"Here are {len(stories)} AI stories you need today.")]
        for s in stories:
            body = _first_words(s.summary, STORY_WORDS[1] - len(s.title.split())) if s.summary else ""
            segments.append(_story_segment(s, f"{s.title.rstrip('.')}. {body}".strip()))
        segments.append(Segment(kind="outro", text=f"Follow for tomorrow's AI news. {SIGN_OFF}"))
        return Episode(
            title=f"AI News Today: {stories[0].headline or stories[0].title}"[:95],
            description=f"Today's top {len(stories)} AI stories in two minutes." + description_footer(stories),
            tags=["AI", "AI news", "artificial intelligence", "tech news", "shorts"],
            segments=segments,
            stories=stories,
        )


def build_writer(llm: LLM | None, show: str, host: str) -> ScriptWriter:
    return LLMWriter(llm, show, host) if llm else TemplateWriter(show, host)


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
