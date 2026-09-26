"""Stage 7: the still layers of the frame: one background per segment, and the news desk.

Layout of the 1080x1920 frame (top to bottom): show header, story card, caption band,
the host, and the desk in front of the host. YouTube's Shorts UI covers roughly the
bottom 300 pixels and a strip on the right, so nothing important sits there.
"""
from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .models import Episode, Segment

W, H = 1080, 1920
CARD = (60, 190, 1020, 830)  # left, top, right, bottom
CAPTION_Y = 950  # centre line of the captions
HOST_Y = 1010  # top of the host sprite
DESK_Y = 1500

NAVY_TOP, NAVY_BOTTOM = (13, 29, 44), (20, 66, 80)
YELLOW = (255, 208, 64)
ORANGE = (238, 118, 30)
INK = (22, 34, 42)
MUTED = (96, 112, 120)
PAPER = (250, 250, 246)
DESK, DESK_EDGE = (31, 111, 120), (18, 70, 78)

BOLD_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]
REGULAR_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def font(size: int, bold: bool = True) -> ImageFont.ImageFont:
    for path in BOLD_FONTS if bold else REGULAR_FONTS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def _gradient() -> Image.Image:
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)
    for y in range(H):
        t = y / H
        d.line([(0, y), (W, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(NAVY_TOP, NAVY_BOTTOM)))
    return img


def _fit(d: ImageDraw.ImageDraw, text: str, max_w: int, max_h: int, sizes: range) -> tuple[str, ImageFont.ImageFont]:
    """Largest font size at which the wrapped text fits the box."""
    for size in sizes:
        f = font(size)
        width_chars = max(8, int(max_w / (size * 0.56)))
        wrapped = textwrap.fill(text, width=width_chars)
        box = d.multiline_textbbox((0, 0), wrapped, font=f, spacing=size // 4)
        if box[2] - box[0] <= max_w and box[3] - box[1] <= max_h:
            return wrapped, f
    f = font(sizes[-1])
    return textwrap.shorten(text, 90, placeholder="…"), f


class StoryCards:
    name = "cards"

    def __init__(self, show: str, day: date | None = None):
        self.show = show
        self.day = day or date.today()

    def _base(self) -> Image.Image:
        img = _gradient()
        d = ImageDraw.Draw(img)
        d.text((60, 82), self.show.upper(), font=font(54), fill=YELLOW)
        stamp = self.day.strftime("%b %d, %Y").upper()
        d.text((W - 60, 92), stamp, font=font(38, bold=False), fill=(200, 214, 220), anchor="ra")
        return img

    def _story(self, seg: Segment, number: int, total: int) -> Image.Image:
        img = self._base()
        d = ImageDraw.Draw(img)
        l, t, r, b = CARD
        d.rounded_rectangle(CARD, radius=36, fill=PAPER)
        d.text((l + 50, t + 44), f"STORY {number} OF {total}", font=font(36), fill=ORANGE)
        # progress dots
        for i in range(total):
            x = r - 50 - (total - 1 - i) * 30
            d.ellipse([x - 9, t + 54, x + 9, t + 72], fill=ORANGE if i < number else (220, 222, 218))
        text, f = _fit(d, seg.headline, r - l - 100, 300, range(78, 44, -4))
        d.multiline_text((l + 50, t + 120), text, font=f, fill=INK, spacing=f.size // 4)
        y = b - 170
        if seg.key_fact:
            chip_font = font(40)
            fact = textwrap.shorten(seg.key_fact, 42, placeholder="…")
            box = d.textbbox((0, 0), fact, font=chip_font)
            d.rounded_rectangle([l + 50, y, l + 90 + box[2], y + 70], radius=35, fill=ORANGE)
            d.text((l + 70, y + 35), fact, font=chip_font, fill=(255, 255, 255), anchor="lm")
        if seg.source:
            d.text((l + 50, b - 56), f"via {seg.source}"[:60], font=font(32, bold=False), fill=MUTED)
        return img

    def _intro(self, episode: Episode) -> Image.Image:
        img = self._base()
        d = ImageDraw.Draw(img)
        l, t, r, b = CARD
        d.rounded_rectangle(CARD, radius=36, fill=PAPER)
        d.text((l + 50, t + 44), "TODAY IN AI", font=font(40), fill=ORANGE)
        y = t + 120
        for i, seg in enumerate(episode.story_segments, 1):
            line = textwrap.shorten(seg.headline, 38, placeholder="…")
            d.text((l + 50, y), f"{i}", font=font(40), fill=ORANGE)
            d.text((l + 110, y), line, font=font(38), fill=INK)
            y += 62
        return img

    def _outro(self) -> Image.Image:
        img = self._base()
        d = ImageDraw.Draw(img)
        l, t, r, b = CARD
        d.rounded_rectangle(CARD, radius=36, fill=PAPER)
        cx = (l + r) // 2
        d.text((cx, t + 200), "Follow for tomorrow's", font=font(64), fill=INK, anchor="mm")
        d.text((cx, t + 290), "AI news", font=font(64), fill=ORANGE, anchor="mm")
        d.text((cx, t + 430), "Sources in the description", font=font(40, bold=False), fill=MUTED, anchor="mm")
        return img

    def render(self, episode: Episode, out_dir: Path) -> list[Path]:
        """One background PNG per segment, in segment order."""
        out_dir.mkdir(parents=True, exist_ok=True)
        total = len(episode.story_segments)
        paths, n = [], 0
        for i, seg in enumerate(episode.segments):
            if seg.kind == "intro":
                img = self._intro(episode)
            elif seg.kind == "outro":
                img = self._outro()
            else:
                n += 1
                img = self._story(seg, n, total)
            path = out_dir / f"card_{i:02d}.png"
            img.save(path)
            paths.append(path)
        return paths

    def desk(self, out_path: Path) -> Path:
        """Transparent full-frame layer with the desk, drawn over the host."""
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([-40, DESK_Y, W + 40, H + 40], radius=40, fill=DESK)
        d.rectangle([0, DESK_Y + 36, W, DESK_Y + 48], fill=DESK_EDGE)
        plate = [W // 2 - 260, DESK_Y + 80, W // 2 + 260, DESK_Y + 180]
        d.rounded_rectangle(plate, radius=24, fill=YELLOW)
        d.text((W // 2, DESK_Y + 130), self.show.upper(), font=font(52), fill=INK, anchor="mm")
        img.save(out_path)
        return out_path
