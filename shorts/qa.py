"""Stage 9: checks that must pass before anything is uploaded."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .media import has_audio, media_duration
from .models import Episode

MAX_SECONDS = 180  # YouTube Shorts limit
MIN_SECONDS = 45


@dataclass
class QAReport:
    passed: bool
    duration: float
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def check(video: Path, episode: Episode, expected_stories: int) -> QAReport:
    problems, warnings = [], []
    duration = media_duration(video)
    if duration > MAX_SECONDS:
        problems.append(f"video is {duration:.0f}s; Shorts must be {MAX_SECONDS}s or less")
    if duration < MIN_SECONDS:
        problems.append(f"video is only {duration:.0f}s")
    if not has_audio(video):
        problems.append("video has no audio track")
    stories = episode.story_segments
    if len(stories) < expected_stories:
        warnings.append(f"{len(stories)} stories instead of {expected_stories}")
    if not stories:
        problems.append("episode has no stories")
    for seg in stories:
        if not seg.url:
            warnings.append(f"no source link for {seg.headline!r}")
    return QAReport(not problems, round(duration, 2), problems, warnings)
