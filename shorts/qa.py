"""Stage 9: checks that must pass before anything is uploaded."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .checks import is_sample
from .media import has_audio, media_duration
from .models import Episode, Voiceover

MAX_SECONDS = 180  # YouTube Shorts limit
MIN_SECONDS = 45


@dataclass
class QAReport:
    passed: bool
    duration: float
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def check(video: Path, episode: Episode, expected_stories: int, voice: Voiceover | None = None,
          allow_sample: bool = False) -> QAReport:
    """``allow_sample`` is for offline runs, which are made of sample stories on purpose."""
    problems, warnings = [], []
    duration = media_duration(video)
    if duration > MAX_SECONDS:
        problems.append(f"video is {duration:.0f}s; Shorts must be {MAX_SECONDS}s or less")
    if duration < MIN_SECONDS:
        problems.append(f"video is only {duration:.0f}s")
    if not has_audio(video):
        problems.append("video has no audio track")
    if voice and voice.silent_segments:
        names = ", ".join("intro" if i == 0 else "outro" if i == len(episode.segments) - 1 else f"story {i}"
                          for i in voice.silent_segments)
        problems.append(f"the voice failed and left silence for: {names}")
    stories = episode.story_segments
    if len(stories) < expected_stories:
        problems.append(f"{len(stories)} stories instead of {expected_stories}")
    if not stories:
        problems.append("episode has no stories")
    if not allow_sample and any(is_sample(s) for s in episode.stories):
        problems.append("the episode contains built-in sample stories, not real news")
    for seg in stories:
        if not seg.url:
            warnings.append(f"no source link for {seg.headline!r}")
    return QAReport(not problems, round(duration, 2), problems, warnings)
