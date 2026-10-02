"""The episode's thumbnail: Quackers at the desk under the day's top headline.

A vertical 1080x1920 PNG, the shape YouTube shows for a Short. It is drawn with the same colours,
fonts and duck as the video, so it costs nothing. The YouTube uploader sets it after the upload; a
hand upload can pick thumbnail.png in Studio.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from PIL import Image, ImageDraw

from .character import draw_duck
from .models import Episode
from .visuals import DESK, DESK_EDGE, DESK_Y, INK, ORANGE, PAPER, YELLOW, H, W, _fit, _gradient, font

HEADLINE_BOX = (60, 400, W - 60, 1000)  # left, top, right, bottom
DUCK_SIZE = 860  # the duck is drawn larger than in the video, so it reads at thumbnail size
DUCK_TOP = 980


def draw_thumbnail(episode: Episode, show: str, day: date | None = None) -> Image.Image:
    day = day or date.today()
    img = _gradient()
    d = ImageDraw.Draw(img)
    d.text((60, 82), show.upper(), font=font(54), fill=YELLOW)
    d.text((W - 60, 92), day.strftime("%b %d, %Y").upper(), font=font(38, bold=False), fill=(200, 214, 220),
           anchor="ra")

    pill_font = font(64)
    label = "TODAY IN AI"
    width = pill_font.getlength(label)
    d.rounded_rectangle([60, 230, 60 + width + 80, 340], radius=55, fill=ORANGE)
    d.text((100, 285), label, font=pill_font, fill=(255, 255, 255), anchor="lm")

    stories = episode.story_segments
    l, t, r, b = HEADLINE_BOX
    if stories:
        text, f = _fit(d, stories[0].headline, r - l, b - t - 110, range(120, 63, -4))
        box = d.multiline_textbbox((l, t), text, font=f, spacing=f.size // 4)
        d.multiline_text((l, t), text, font=f, fill=PAPER, spacing=f.size // 4)
        if len(stories) > 1:
            more = f"+ {len(stories) - 1} more {'story' if len(stories) == 2 else 'stories'}"
            d.text((l, box[3] + 40), more, font=font(60), fill=YELLOW)

    duck = draw_duck(2, True).resize((DUCK_SIZE, DUCK_SIZE), Image.LANCZOS)
    img.paste(duck, ((W - DUCK_SIZE) // 2, DUCK_TOP), duck)
    desk = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    _desk(ImageDraw.Draw(desk), show)
    img.paste(desk, (0, 0), desk)
    return img


def _desk(d: ImageDraw.ImageDraw, show: str) -> None:
    """The same desk as the video's, drawn a little lower so more of the duck shows."""
    top = DESK_Y + 120
    d.rounded_rectangle([-40, top, W + 40, H + 40], radius=40, fill=DESK)
    d.rectangle([0, top + 36, W, top + 48], fill=DESK_EDGE)
    d.rounded_rectangle([W // 2 - 260, top + 80, W // 2 + 260, top + 180], radius=24, fill=YELLOW)
    d.text((W // 2, top + 130), show.upper(), font=font(52), fill=INK, anchor="mm")


def save_thumbnail(episode: Episode, show: str, out_path: Path, day: date | None = None) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    draw_thumbnail(episode, show, day).save(out_path, optimize=True)
    return out_path
