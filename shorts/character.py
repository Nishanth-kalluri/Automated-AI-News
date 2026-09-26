"""Stage 6: animate the host.

An animator turns the voiceover into a ``CharacterTrack``: ffmpeg input arguments for a
video layer that the composer places in the host area. ``PuppetDuck`` draws the duck with
Pillow and flaps its bill to the loudness of the voice, which costs nothing. A generated
talking clip (for example Kling AI Avatar) fits the same interface.
"""
from __future__ import annotations

import logging
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw

from .media import ffmpeg_exe
from .models import Voiceover

log = logging.getLogger(__name__)
ANIM_FPS = 15
SPRITE = 640  # sprite canvas is SPRITE x SPRITE pixels

YELLOW = (255, 208, 64)
YELLOW_DARK = (236, 170, 30)
BILL = (255, 132, 38)
BILL_DARK = (214, 92, 20)
MOUTH = (120, 30, 40)
TONGUE = (240, 110, 120)
INK = (48, 34, 30)
CHEEK = (255, 150, 150)
TIE = (31, 111, 120)


@dataclass
class CharacterTrack:
    input_args: list[str]  # ffmpeg input arguments for the layer
    width: int
    height: int
    bob: bool = True  # composer adds a gentle idle bob
    files: list[Path] = field(default_factory=list)


class Animator(Protocol):
    name: str

    def animate(self, voice: Voiceover, work_dir: Path) -> CharacterTrack: ...


def draw_duck(mouth: int, eyes_open: bool) -> Image.Image:
    """mouth: 0 closed, 1 half open, 2 open."""
    img = Image.new("RGBA", (SPRITE, SPRITE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ow = 9  # outline width
    # body and wings
    d.ellipse([110, 330, 530, 700], fill=YELLOW, outline=INK, width=ow)
    d.ellipse([70, 420, 190, 560], fill=YELLOW_DARK, outline=INK, width=ow)
    d.ellipse([450, 420, 570, 560], fill=YELLOW_DARK, outline=INK, width=ow)
    # head with a tuft
    d.ellipse([140, 60, 500, 400], fill=YELLOW, outline=INK, width=ow)
    d.arc([290, 20, 360, 110], 180, 330, fill=INK, width=ow)
    d.arc([315, 30, 385, 100], 190, 320, fill=INK, width=ow)
    # bow tie
    d.polygon([(320, 405), (250, 370), (250, 440)], fill=TIE, outline=INK)
    d.polygon([(320, 405), (390, 370), (390, 440)], fill=TIE, outline=INK)
    d.ellipse([302, 387, 338, 423], fill=TIE, outline=INK, width=5)
    # cheeks
    d.ellipse([170, 250, 225, 285], fill=CHEEK)
    d.ellipse([415, 250, 470, 285], fill=CHEEK)
    # eyes
    for cx in (250, 390):
        if eyes_open:
            d.ellipse([cx - 30, 160, cx + 30, 240], fill=INK)
            d.ellipse([cx - 14, 172, cx + 4, 192], fill=(255, 255, 255))
        else:
            d.arc([cx - 30, 180, cx + 30, 230], 200, 340, fill=INK, width=ow)
    # bill
    gap = (0, 26, 56)[mouth]
    d.rounded_rectangle([230, 255, 410, 305], radius=24, fill=BILL, outline=INK, width=ow)
    if gap:
        d.rounded_rectangle([248, 290, 392, 300 + gap], radius=20, fill=MOUTH, outline=INK, width=6)
        if mouth == 2:
            d.ellipse([285, 300 + gap - 26, 355, 300 + gap + 6], fill=TONGUE)
    d.rounded_rectangle([244, 298 + gap, 396, 332 + gap], radius=18, fill=BILL_DARK, outline=INK, width=ow)
    d.line([270, 277, 290, 277], fill=INK, width=5)  # nostrils
    d.line([350, 277, 370, 277], fill=INK, width=5)
    return img


def loudness_envelope(audio: Path, fps: int = ANIM_FPS) -> list[float]:
    """RMS loudness of the audio per animation frame, 0..1 scaled to the loud parts."""
    rate = 16000
    raw = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", str(audio), "-f", "s16le", "-ac", "1",
                          "-ar", str(rate), "-"], capture_output=True, check=True).stdout
    samples = memoryview(raw).cast("h")
    step = rate // fps
    env = []
    for i in range(0, len(samples), step):
        chunk = samples[i:i + step]
        env.append((sum(s * s for s in chunk) / max(len(chunk), 1)) ** 0.5)
    loud = sorted(env)[int(len(env) * 0.9)] if env else 0
    return [min(e / loud, 1.0) if loud else 0.0 for e in env]


def mouth_states(envelope: list[float]) -> list[int]:
    return [0 if e < 0.15 else 1 if e < 0.5 else 2 for e in envelope]


def blink_frames(n: int, seed: int, fps: int = ANIM_FPS) -> set[int]:
    rng = random.Random(seed)
    frames, t = set(), rng.uniform(1.5, 3.0)
    while t * fps < n:
        start = int(t * fps)
        frames.update({start, start + 1})
        t += rng.uniform(2.5, 5.0)
    return frames


def run_length(states: list[tuple[int, bool]], fps: int) -> list[tuple[tuple[int, bool], float]]:
    out: list[tuple[tuple[int, bool], float]] = []
    for s in states:
        if out and out[-1][0] == s:
            out[-1] = (s, out[-1][1] + 1 / fps)
        else:
            out.append((s, 1 / fps))
    return out


class PuppetDuck:
    name = "puppet"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def animate(self, voice: Voiceover, work_dir: Path) -> CharacterTrack:
        work_dir.mkdir(parents=True, exist_ok=True)
        sprites: dict[tuple[int, bool], Path] = {}
        for mouth in (0, 1, 2):
            for eyes in (True, False):
                path = work_dir / f"duck_m{mouth}_{'open' if eyes else 'shut'}.png"
                draw_duck(mouth, eyes).save(path)
                sprites[(mouth, eyes)] = path
        mouths = mouth_states(loudness_envelope(voice.audio_path))
        blinks = blink_frames(len(mouths), self.seed)
        states = [(m, i not in blinks) for i, m in enumerate(mouths)]
        lines = [f"file '{sprites[s].resolve()}'\nduration {d:.4f}\n" for s, d in run_length(states, ANIM_FPS)]
        lines.append(f"file '{sprites[states[-1]].resolve()}'\n")  # concat demuxer needs the last file twice
        listing = work_dir / "duck.txt"
        listing.write_text("".join(lines))
        return CharacterTrack(["-f", "concat", "-safe", "0", "-i", str(listing.resolve())], SPRITE, SPRITE,
                              files=list(sprites.values()))


def build_animator(name: str, seed: int = 0) -> Animator:
    if name == "puppet":
        return PuppetDuck(seed)
    raise ValueError(f"Unknown animator {name!r}. Known: puppet")
