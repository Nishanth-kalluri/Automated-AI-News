"""The outro: a subscribe line that changes every day, then the show's fixed sign-off.

Each line asks viewers to subscribe with a playful threat of what happens if they don't, in the spirit
of "don't get lost in the storm of AI news": the storm, being the last to know, or the duck itself.
The lines rotate in order, and ``OutroLog`` on the state branch remembers the last ones used so a line
comes back only after all the others have aired. ``SHORTS_OUTRO`` replaces the rotation with one fixed outro.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

SIGN_OFF = "That's the news from the pond. See you tomorrow!"
# In rotation order: storm lines lead, and lines with the same image are kept apart.
SUBSCRIBE_LINES = (
    "Hit subscribe, or get swept into the AI news storm. I don't paddle back for stragglers.",
    "If you don't subscribe, every duck in this pond will know. We talk.",
    "Picture yourself in the next AI storm with no duck and no umbrella. Now go subscribe.",
    "At the next family dinner, your nephew will explain AI to you. Subscribe before that happens.",
    "Forecast: heavy AI news, high chance of flooding. Subscribe to stay dry.",
    "Consider this your first and final warning. Subscribe, or receive one very strongly worded honk.",
    "AI news rolls off me like water off a duck's back. You? Not so much. Subscribe.",
    "Unless you enjoy being a week behind your own group chat, subscribe.",
    "Scroll away without subscribing, and that whole AI flood is yours to mop up alone.",
    "Don't make me waddle over there. Subscribe, and these little legs can stay right here.",
    "Without this duck at the wheel, you're adrift in an ocean of AI headlines. Better subscribe.",
    "When your boss mentions the big AI launch, don't be the one asking which launch. Subscribe.",
    "Another tidal wave of AI news breaks at sunrise, ready or not. Subscribe and ride it out.",
    "I keep a list of everyone who didn't subscribe. It's in crayon, and it's getting long.",
    "Your socks won't survive another AI flood like today's. Do yourself a favor and subscribe.",
    "Subscribe now, or you'll learn today's AI news from a meme a month late.",
    "Last dry seat on my boat out of the AI storm. Subscribe and it's yours.",
    "One more video without subscribing, and I shake pond water all over your screen.",
    "Ride the AI wave with me or let it roll over you. The subscribe button decides.",
    "Without me, every coffee break becomes an AI pop quiz you never studied for. Subscribe.",
    "Somewhere, someone just got splashed by an AI launch they never saw coming. Subscribe so it's never you.",
    "Every day you don't subscribe, I quack outside your window, loudly and slightly off key.",
    "Don't be stuck wringing out last week's AI news. Subscribe, and I'll keep you current.",
    "Skip a day and you'll miss a whole new batch of AI buzzwords. Subscribe and stay fluent.",
)


def outro_text(line: str) -> str:
    return f"{line} {SIGN_OFF}"


class OutroLog:
    """The last outros that aired, so the next one is the next line in the rotation."""

    KEEP = len(SUBSCRIBE_LINES)

    def __init__(self, path: Path):
        self.path = path
        try:
            entries = json.loads(path.read_text()) if path.exists() else []
        except (OSError, ValueError):
            entries = []
        self.entries: list[dict] = ([e for e in entries if isinstance(e, dict) and isinstance(e.get("text"), str)]
                                    if isinstance(entries, list) else [])

    def _line(self, text: str) -> int | None:
        return next((i for i, line in enumerate(SUBSCRIBE_LINES) if text.startswith(line)), None)

    def next(self) -> str:
        """The outro after the last one that aired, skipping any line used in the last ``KEEP`` outros."""
        used = [i for i in (self._line(e["text"]) for e in self.entries) if i is not None]
        start = used[-1] + 1 if used else 0
        recent = set(used[-(len(SUBSCRIBE_LINES) - 1):]) if len(SUBSCRIBE_LINES) > 1 else set()
        order = [(start + k) % len(SUBSCRIBE_LINES) for k in range(len(SUBSCRIBE_LINES))]
        pick = next((i for i in order if i not in recent), order[0])
        return outro_text(SUBSCRIBE_LINES[pick])

    def add(self, text: str) -> None:
        self.entries = [*self.entries, {"date": date.today().isoformat(), "text": text}][-self.KEEP:]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=1))
