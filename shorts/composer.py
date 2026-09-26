"""Stage 8: put the layers on a timeline and render the final video with one ffmpeg call.

Layers, bottom to top: segment backgrounds, the host, the desk, word-by-word captions.
The timeline is written to timeline.json first so a different renderer (Remotion, a
cloud editor) can consume the same description later.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .character import CharacterTrack
from .media import run_ffmpeg
from .models import Voiceover, Word
from .visuals import CAPTION_Y, HOST_Y, W

FPS = 30
WORDS_PER_CAPTION = 3


def _ass_time(t: float) -> str:
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def caption_chunks(words: list[Word], size: int = WORDS_PER_CAPTION) -> list[list[Word]]:
    """Groups of up to ``size`` words that never run across a sentence end."""
    chunks: list[list[Word]] = []
    current: list[Word] = []
    for w in words:
        current.append(w)
        if len(current) == size or re.search(r"[.!?]$", w.text):
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def captions_ass(words: list[Word]) -> str:
    """Karaoke-style captions: each word turns yellow as it is spoken."""
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,DejaVu Sans,76,&H0040D0FF,&H00FFFFFF,&H00201810,&H80000000,-1,0,0,0,100,100,0,0,1,7,2,5,60,60,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for chunk in caption_chunks(words):
        start, end = chunk[0].start, chunk[-1].end
        karaoke = " ".join(
            f"{{\\k{max(int(round((w.end - w.start) * 100)), 1)}}}{w.text.replace('{', '(').replace('}', ')')}"
            for w in chunk)
        lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Cap,,0,0,0,,"
                     f"{{\\pos({W // 2},{CAPTION_Y})}}{karaoke}")
    return header + "\n".join(lines) + "\n"


def background_concat(cards: list[Path], voice: Voiceover) -> str:
    """ffmpeg concat list: each card shows from its segment's start to the next segment's start."""
    starts = [s for s, _ in voice.segment_timings]
    ends = starts[1:] + [voice.duration]
    lines = [f"file '{c.resolve()}'\nduration {max(e - s, 0.04):.3f}\n" for c, s, e in zip(cards, starts, ends)]
    lines.append(f"file '{cards[-1].resolve()}'\n")
    return "".join(lines)


def render(voice: Voiceover, cards: list[Path], desk: Path, host: CharacterTrack,
           out_path: Path, x264_preset: str = "medium") -> Path:
    work = out_path.parent
    (work / "backgrounds.txt").write_text(background_concat(cards, voice))
    (work / "captions.ass").write_text(captions_ass(voice.words))
    host_x = (W - host.width) // 2
    host_y = f"{HOST_Y}+12*sin(2*PI*t/1.8)" if host.bob else str(HOST_Y)
    timeline = {
        "fps": FPS, "size": [1080, 1920], "duration": voice.duration,
        "segments": [{"start": s, "end": e, "background": c.name}
                     for (s, e), c in zip(voice.segment_timings, cards)],
        "layers": ["backgrounds.txt", {"host": host.input_args, "x": host_x, "y": host_y},
                   desk.name, "captions.ass"],
        "audio": voice.audio_path.name,
    }
    (work / "timeline.json").write_text(json.dumps(timeline, indent=2))

    graph = (
        f"[0:v]fps={FPS},format=rgba[bg];"
        f"[1:v]fps={FPS},format=rgba[host];"
        f"[bg][host]overlay=x={host_x}:y='{host_y}':eof_action=repeat[v1];"
        f"[v1][2:v]overlay=0:0[v2];"
        f"[v2]ass=captions.ass,format=yuv420p[v]"
    )
    run_ffmpeg([
        "-f", "concat", "-safe", "0", "-i", "backgrounds.txt",
        *host.input_args,
        "-loop", "1", "-framerate", str(FPS), "-i", str(desk.resolve()),
        "-i", str(voice.audio_path.resolve()),
        "-filter_complex", graph, "-map", "[v]", "-map", "3:a",
        "-t", f"{voice.duration:.3f}", "-r", str(FPS),
        "-c:v", "libx264", "-preset", x264_preset, "-crf", "21",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-movflags", "+faststart", str(out_path.resolve()),
    ], cwd=work)
    return out_path
