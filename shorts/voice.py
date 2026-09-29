"""Stage 5: voice every segment and time each word for the captions.

``lineup`` reads a finished script with several voices, one audio file each, so voices can be
compared on the same words before picking one with SHORTS_VOICE / SHORTS_EDGE_VOICE / SHORTS_OPENAI_VOICE.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
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


class OpenAIVoice:
    """OpenAI text to speech (SHORTS_VOICE=openai): more expressive, about 1.5 cents per minute of speech.
    The instructions set the character and pace."""

    name = "openai"

    def __init__(self, api_key: str, voice: str, model: str, instructions: str = "", client=None):
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self.client, self.voice, self.model, self.instructions = client, voice, model, instructions

    def speak(self, text: str, out_path: Path) -> Path:
        path = out_path.with_suffix(".mp3")
        kwargs = {"model": self.model, "voice": self.voice, "input": text, "response_format": "mp3"}
        if self.instructions:
            kwargs["instructions"] = self.instructions
        path.write_bytes(self.client.audio.speech.create(**kwargs).content)
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


def _openai_voice(cfg: Config, voice: str) -> OpenAIVoice:
    if not cfg.openai_api_key:
        raise ValueError("the openai voice needs OPENAI_API_KEY")
    return OpenAIVoice(cfg.openai_api_key, voice, cfg.openai_tts_model,
                       cfg.openai_tts_instructions.replace("{host}", cfg.host_name))


def build_voice(cfg: Config) -> VoiceProvider:
    if cfg.voice == "edge":
        try:
            return EdgeVoice(cfg.edge_voice, cfg.edge_rate)
        except ImportError:
            log.warning("edge-tts not installed; using silent voice")
    elif cfg.voice == "openai":
        return _openai_voice(cfg, cfg.openai_voice)
    elif cfg.voice != "silent":
        raise ValueError(f"Unknown voice provider {cfg.voice!r}")
    return SilentVoice()


# The standard comparison set: free Microsoft voices (the current one first, then two other child-like
# voices and some lively adult ones), then OpenAI voices that take acting instructions.
LINEUP_EDGE = ["en-US-AnaNeural", "en-GB-MaisieNeural", "en-US-AvaMultilingualNeural",
               "en-US-EmmaMultilingualNeural", "en-US-AndrewMultilingualNeural", "en-US-BrianMultilingualNeural",
               "en-US-JennyNeural", "en-US-AriaNeural", "en-US-GuyNeural", "en-AU-NatashaNeural"]
LINEUP_OPENAI = ["coral", "nova", "fable", "shimmer"]


def parse_lineup(specs: list[str]) -> list[tuple[str, str]]:
    """SHORTS_VOICE_LINEUP -> [(provider, voice)]. "all" is the standard set, "edge" or "openai" its half;
    otherwise "edge:en-US-AnaNeural", "openai:coral", or a bare name (Microsoft names contain "Neural")."""
    out: list[tuple[str, str]] = []
    for spec in (x.strip() for x in specs):
        low = spec.lower()
        if low in ("all", "default", "standard", "on", "yes", "true"):
            items = [("edge", v) for v in LINEUP_EDGE] + [("openai", v) for v in LINEUP_OPENAI]
        elif low == "edge":
            items = [("edge", v) for v in LINEUP_EDGE]
        elif low == "openai":
            items = [("openai", v) for v in LINEUP_OPENAI]
        elif ":" in spec:
            provider, _, name = spec.partition(":")
            items = [(provider.strip().lower(), name.strip())]
        elif spec:
            items = [("edge" if "neural" in low else "openai", spec)]
        else:
            items = []
        out += [i for i in items if i not in out and i[1]]
    return out


def _setting(provider: str, name: str) -> str:
    return (f"SHORTS_VOICE=edge and SHORTS_EDGE_VOICE={name}" if provider == "edge"
            else f"SHORTS_VOICE=openai and SHORTS_OPENAI_VOICE={name}")


def lineup(cfg: Config, episode: Episode, out_dir: Path, voices: list[tuple[str, str]] | None = None) -> list[dict]:
    """Read ``episode`` with every lineup voice into ``out_dir``: one MP3 per voice plus voices.txt, which
    says how to pick one. A voice that fails is listed with the error; this never raises."""
    voices = parse_lineup(cfg.voice_lineup) if voices is None else voices
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, (provider, name) in enumerate(voices, 1):
        label = f"{k:02d}-{provider}-{re.sub(r'[^A-Za-z0-9-]+', '-', name)}"
        row = {"voice": f"{provider}:{name}", "setting": _setting(provider, name)}
        work = out_dir / f".{label}"
        try:
            if provider == "edge":
                voice: VoiceProvider = EdgeVoice(name, cfg.edge_rate)
            elif provider == "openai":
                voice = _openai_voice(cfg, name)
            else:
                raise ValueError(f"unknown voice provider {provider!r}")
            vo = narrate(voice, episode, work)
            if vo.silent_segments:
                raise RuntimeError(f"the voice failed on {len(vo.silent_segments)} of {len(episode.segments)} parts")
            mp3 = out_dir / f"{label}.mp3"
            run_ffmpeg(["-i", str(vo.audio_path), "-codec:a", "libmp3lame", "-b:a", "128k", str(mp3)])
            row.update(file=mp3.name, seconds=round(vo.duration, 1))
            log.info("      %s: %.0fs", row["voice"], vo.duration)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
            log.warning("      %s failed: %s", row["voice"], row["error"])
        finally:
            shutil.rmtree(work, ignore_errors=True)
        rows.append(row)
    (out_dir / "voices.json").write_text(json.dumps(rows, indent=1))
    lines = ["Every file reads the same script. To use a voice, set the repository variables shown.", ""]
    for r in rows:
        result = r["error"] if "error" in r else f"{r['seconds']:.0f} seconds"
        lines += [f"{r.get('file') or r['voice']}: {result}", f"    {r['setting']}"]
    (out_dir / "voices.txt").write_text("\n".join(lines) + "\n")
    return rows


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
