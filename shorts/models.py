from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Evidence:
    quote: str  # copied verbatim from ``url``, and checked in code to be there
    url: str


@dataclass
class Story:
    title: str
    url: str
    source: str
    published: datetime
    summary: str = ""
    popularity: float = 0.0  # source-provided, 0..1 (e.g. HN points)
    score: float = 0.0  # set by the heuristic editor
    kind: str = "article"  # "article", "newsletter" (a whole issue covering many stories), or "web" (a search hit)
    body: str = ""  # full text: the newsletter issue, or the article once researched
    # Filled in by the editor for picked stories.
    headline: str = ""  # short headline for the on-screen card
    key_fact: str = ""  # one number or fact worth showing on screen
    outlets: list[str] = field(default_factory=list)  # every source that covered it
    # Filled in by the research agents (SHORTS_WEB=on).
    evidence: list[Evidence] = field(default_factory=list)
    first_reported: str = ""  # YYYY-MM-DD, only when a fetched page backs it
    checked: str = ""  # "", verified, thin, stale, wrong_story or failed


@dataclass
class Segment:
    kind: str  # "intro", "story" or "outro"
    text: str  # what the host says
    headline: str = ""
    key_fact: str = ""
    source: str = ""
    url: str = ""


@dataclass
class Episode:
    title: str
    description: str
    tags: list[str]
    segments: list[Segment]
    stories: list[Story] = field(default_factory=list)

    @property
    def narration(self) -> str:
        return " ".join(s.text for s in self.segments)

    @property
    def story_segments(self) -> list[Segment]:
        return [s for s in self.segments if s.kind == "story"]


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Voiceover:
    audio_path: Path
    duration: float
    # (start, end) seconds per Episode segment, same order as Episode.segments.
    segment_timings: list[tuple[float, float]]
    words: list[Word] = field(default_factory=list)
    # Segments where the voice failed and silence was put in instead (0-based segment index).
    silent_segments: list[int] = field(default_factory=list)


@dataclass
class UploadResult:
    uploader: str
    location: str
    note: str = ""  # something the email should say, such as a thumbnail YouTube refused
