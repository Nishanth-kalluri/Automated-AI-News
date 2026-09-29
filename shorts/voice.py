"""Stage 5: voice every segment and time each word for the captions."""
from __future__ import annotations

import asyncio
import logging
import wave
from pathlib import Path
from typing import Protocol

from .config import Config
from .media import media_duration, run_ffmpeg
from .models import Episode, Voiceover, Word

log = logging.getLogger(__name__)
WORDS_PER_SECOND = 2.6  # ~155 wpm, typical Shorts narration pace
GAP = 0.3  # seconds of silence between segments
SAMPLE_RATE = 24000


class VoiceProvider(Protocol):
    name: str

    def speak(self, text: str, out_path: Path) -> Path: ...


def word_timings(text: str, start: float, end: float) -> list[Word]:
    """Spread words over [start, end] by length. Close enough for 3-word caption chunks."""
    words = text.split()
    weights = [len(w) + 2 for w in words]
    total, t, out = sum(weights) or 1, start, []
    for w, weight in zip(words, weights):
        d = (end - start) * weight / total
        out.append(Word(w, t, t + d))
        t += d
    return out


class EdgeVoice:
    """Free Microsoft neural TTS via edge-tts. Needs internet, no API key."""

    name = "edge"

    def __init__(self, voice: str, rate: str):
        import edge_tts  # noqa: F401  (fail early if missing)

        self.voice, self.rate = voice, rate

    def speak(self, text: str, out_path: Path) -> Path:
        import edge_tts

        path = out_path.with_suffix(".mp3")
        asyncio.run(edge_tts.Communicate(text, self.voice, rate=self.rate).save(str(path)))
        return path


class SilentVoice:
    """Offline stub: silence paced like real narration, so video timing is realistic."""

    name = "silent"

    def speak(self, text: str, out_path: Path) -> Path:
        duration = max(len(text.split()) / WORDS_PER_SECOND, 1.0)
        path = out_path.with_suffix(".wav")
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(b"\x00\x00" * int(SAMPLE_RATE * duration))
        return path


def build_voice(cfg: Config) -> VoiceProvider:
    if cfg.voice == "edge":
        try:
            return EdgeVoice(cfg.edge_voice, cfg.edge_rate)
        except ImportError:
            log.warning("edge-tts not installed; using silent voice")
    elif cfg.voice != "silent":
        raise ValueError(f"Unknown voice provider {cfg.voice!r}")
    return SilentVoice()


def _speak(voice: VoiceProvider, text: str, out_path: Path) -> tuple[Path, bool]:
    """The clip, and whether it had to fall back to silence."""
    try:
        return voice.speak(text, out_path), False
    except Exception as exc:
        if isinstance(voice, SilentVoice):
            raise
        log.warning("%s voice failed (%s); using silence for this segment", voice.name, exc)
        return SilentVoice().speak(text, out_path), True


def _cached_clip(voice: VoiceProvider, text: str, out_path: Path) -> Path | None:
    """A clip from an earlier pass of this run with the same voice and text (after a trim, most are)."""
    note = out_path.with_suffix(".txt")
    if note.exists() and note.read_text() == f"{voice.name}\n{text}":
        return next((p for p in out_path.parent.glob(out_path.name + ".*") if p.suffix in (".mp3", ".wav")), None)
    return None


def narrate(voice: VoiceProvider, episode: Episode, out_dir: Path) -> Voiceover:
    out_dir.mkdir(parents=True, exist_ok=True)
    clips, silent = [], []
    for i, seg in enumerate(episode.segments):
        out_path = out_dir / f"seg_{i:02d}"
        clip = _cached_clip(voice, seg.text, out_path)
        if clip is None:
            for old in out_dir.glob(out_path.name + ".*"):
                old.unlink()
            clip, fell_back = _speak(voice, seg.text, out_path)
            if fell_back:
                silent.append(i)
            else:
                out_path.with_suffix(".txt").write_text(f"{voice.name}\n{seg.text}")
        clips.append(clip)
    durations = [media_duration(c) for c in clips]

    # Resample every clip to one format, pad a short pause after each, and join.
    inputs: list[str] = []
    chains = []
    for i, clip in enumerate(clips):
        inputs += ["-i", str(clip)]
        pad = f",apad=pad_dur={GAP}" if i < len(clips) - 1 else ""
        chains.append(f"[{i}:a]aresample={SAMPLE_RATE},aformat=channel_layouts=mono{pad}[a{i}]")
    graph = ";".join(chains) + ";" + "".join(f"[a{i}]" for i in range(len(clips))) + \
        f"concat=n={len(clips)}:v=0:a=1[out]"
    audio = out_dir / "voice.wav"
    run_ffmpeg([*inputs, "-filter_complex", graph, "-map", "[out]", str(audio)])

    timings, words, t = [], [], 0.0
    for seg, d in zip(episode.segments, durations):
        timings.append((t, t + d))
        words += word_timings(seg.text, t, t + d)
        t += d + GAP
    return Voiceover(audio, media_duration(audio), timings, words, silent)
