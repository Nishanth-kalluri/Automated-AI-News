"""Background music: one track a day from a folder, mixed quietly under the voice by the composer.

Put a few tracks in assets/music/ (or SHORTS_MUSIC_DIR); each day's episode takes the next one in
name order. With no tracks the video has the voice only.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)
AUDIO = (".mp3", ".m4a", ".aac", ".wav", ".ogg", ".flac")


def tracks(folder: Path | None) -> list[Path]:
    if folder is None or not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO)


def pick_track(folder: Path | None, day: date | None = None) -> Path | None:
    """Today's track: consecutive days take consecutive tracks, so each one airs as rarely as possible."""
    found = tracks(folder)
    if not found:
        return None
    return found[(day or date.today()).toordinal() % len(found)]
