from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from .outro import SUBSCRIBE_LINES, outro_text

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional at runtime
    pass

DEFAULT_RSS_FEEDS = [
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "https://venturebeat.com/category/ai/feed/",
    "https://www.technologyreview.com/topic/artificial-intelligence/feed",
    "https://feeds.arstechnica.com/arstechnica/technology-lab",
    "https://openai.com/news/rss.xml",
    "https://blog.google/technology/ai/rss/",
    "https://huggingface.co/blog/feed.xml",
]
# The outro when none is picked: the rotation's first subscribe line and the sign-off. Runs use the daily
# rotation (outro.OutroLog) unless SHORTS_OUTRO sets one fixed outro.
DEFAULT_OUTRO = outro_text(SUBSCRIBE_LINES[0])
YOUTUBE_PRIVACY = ("private", "unlisted", "public")
DEFAULT_OPENAI_TTS_INSTRUCTIONS = ("You are {host}, a small, cheerful cartoon duck who anchors a daily AI news show. "
                                   "Sound bright, playful and warm, with an upbeat, quick pace and clear diction.")
DEFAULT_LLM_MODELS = {"openai": "gpt-5", "anthropic": "claude-opus-5"}
# Per-role defaults when SHORTS_LLM_MODEL isn't set. A role model the account can't use falls
# back to the base model above on the first call.
DEFAULT_ROLE_MODELS = {"openai": {"editor": "gpt-6-sol", "writer": "gpt-6-sol", "checker": "gpt-6-luna"}}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


def _on(name: str, default: str) -> bool:
    return _env(name, default).lower() not in ("off", "0", "false", "no")


def _privacy(value: str) -> str:
    """SHORTS_YOUTUBE_PRIVACY, checked when the run starts rather than rejected by YouTube at the end."""
    value = value.lower()
    if value not in YOUTUBE_PRIVACY:
        raise ValueError(f"SHORTS_YOUTUBE_PRIVACY must be one of {', '.join(YOUTUBE_PRIVACY)}, not {value!r}")
    return value


def _volume(value: str) -> float:
    """SHORTS_MUSIC_VOLUME, checked when the run starts: 0.15 is quiet, 1 is as loud as the track."""
    volume = float(value)
    if not 0 <= volume <= 1:
        raise ValueError(f"SHORTS_MUSIC_VOLUME must be between 0 and 1, not {value!r}")
    return volume


def _list(name: str, default: list[str]) -> list[str]:
    raw = _env(name)
    return [x.strip() for x in raw.split(",") if x.strip()] if raw else list(default)


@dataclass
class Config:
    sources: list[str]
    rss_feeds: list[str]
    stories_per_video: int  # at most; fewer when there aren't enough solid stories
    min_stories: int  # below this, the day is skipped rather than padded with weak stories
    max_age_hours: float
    # LLM that picks stories and writes the script: openai, anthropic, or none (templates).
    llm_provider: str
    llm_model: str
    # Without an LLM the script would be feed text read out by a template: skipped unless this is on.
    allow_no_ai: bool
    # Agent loops (repair rounds, tool use) on top of the LLM stages; off means one-shot calls.
    agents: bool
    editor_model: str
    writer_model: str
    checker_model: str
    max_repairs: int  # repair rounds per agent loop
    budget_usd: float  # per run
    monthly_budget_usd: float
    # Phase 2: research agents with web tools, and the shadow run that compares them with the live path.
    web: bool
    shadow: bool
    tavily_monthly_credits: int
    tavily_run_credits: int
    openai_api_key: str
    anthropic_api_key: str
    # Newsletter inbox (AgentMail) and article reader (Tavily).
    agentmail_api_key: str
    agentmail_inbox: str
    tavily_api_key: str
    show_name: str
    host_name: str
    outro: str  # the last line, subscribe ask and sign-off: SHORTS_OUTRO, else today's from the rotation
    voice: str  # edge, openai or silent
    edge_voice: str
    edge_rate: str
    openai_voice: str
    openai_tts_model: str
    openai_tts_instructions: str
    # Extra voices to read the finished script with, for comparing voices ("all" for the standard set).
    voice_lineup: list[str]
    animator: str
    # Background music: a folder of tracks, one a day, quiet under the voice. None (SHORTS_MUSIC=off): voice only.
    music_dir: Path | None
    music_volume: float  # the music's level while nobody speaks, 0 to 1; it drops further under the voice
    x264_preset: str
    uploader: str
    youtube_privacy: str
    # Days a YouTube sign-in lasts: 7 while the Google Cloud app is in Testing, 0 once it is published.
    youtube_signin_days: int
    notify_email: str
    output_dir: Path
    state_dir: Path

    @classmethod
    def from_env(cls) -> "Config":
        openai_key, anthropic_key = _env("OPENAI_API_KEY"), _env("ANTHROPIC_API_KEY")
        auto = "openai" if openai_key else "anthropic" if anthropic_key else "none"
        provider = _env("SHORTS_LLM_PROVIDER", auto)
        model = _env("SHORTS_LLM_MODEL", DEFAULT_LLM_MODELS.get(provider, ""))
        roles = {} if _env("SHORTS_LLM_MODEL") else DEFAULT_ROLE_MODELS.get(provider, {})
        return cls(
            # Hacker News and Reddit are discussion sites, not news; they're opt-in. Hacker News still
            # helps rank stories through the coverage lookup when SHORTS_WEB is on.
            sources=_list("SHORTS_SOURCES", ["newsletter", "rss"]),
            rss_feeds=_list("SHORTS_RSS_FEEDS", DEFAULT_RSS_FEEDS),
            stories_per_video=int(_env("SHORTS_STORIES_PER_VIDEO", "8")),
            min_stories=int(_env("SHORTS_MIN_STORIES", "4")),
            max_age_hours=float(_env("SHORTS_MAX_AGE_HOURS", "30")),
            llm_provider=provider,
            llm_model=model,
            allow_no_ai=_on("SHORTS_ALLOW_NO_AI", "off"),
            agents=_on("SHORTS_AGENTS", "on"),
            editor_model=_env("SHORTS_EDITOR_MODEL", roles.get("editor", model)),
            writer_model=_env("SHORTS_WRITER_MODEL", roles.get("writer", model)),
            checker_model=_env("SHORTS_CHECKER_MODEL", roles.get("checker", model)),
            max_repairs=int(_env("SHORTS_MAX_REPAIRS", "2")),
            budget_usd=float(_env("SHORTS_BUDGET_USD", "0.60")),
            monthly_budget_usd=float(_env("SHORTS_MONTHLY_BUDGET_USD", "18")),
            web=_on("SHORTS_WEB", "off"),
            shadow=_on("SHORTS_SHADOW", "off"),
            tavily_monthly_credits=int(_env("SHORTS_TAVILY_MONTHLY_CREDITS", "700")),
            tavily_run_credits=int(_env("SHORTS_TAVILY_RUN_CREDITS", "25")),
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
            agentmail_api_key=_env("AGENTMAIL_API_KEY"),
            agentmail_inbox=_env("AGENTMAIL_INBOX"),
            tavily_api_key=_env("TAVILY_API_KEY"),
            show_name=_env("SHORTS_SHOW_NAME", "Duck Desk"),
            host_name=_env("SHORTS_HOST_NAME", "Quackers"),
            outro=_env("SHORTS_OUTRO"),  # empty: the daily rotation, picked when a run starts
            voice=_env("SHORTS_VOICE", "edge"),
            edge_voice=_env("SHORTS_EDGE_VOICE", "en-US-AnaNeural"),
            edge_rate=_env("SHORTS_EDGE_RATE", "+18%"),
            openai_voice=_env("SHORTS_OPENAI_VOICE", "coral"),
            openai_tts_model=_env("SHORTS_OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
            openai_tts_instructions=_env("SHORTS_OPENAI_TTS_INSTRUCTIONS", DEFAULT_OPENAI_TTS_INSTRUCTIONS),
            voice_lineup=_list("SHORTS_VOICE_LINEUP", []),
            animator=_env("SHORTS_ANIMATOR", "puppet"),
            music_dir=Path(_env("SHORTS_MUSIC_DIR", "assets/music")) if _on("SHORTS_MUSIC", "on") else None,
            music_volume=_volume(_env("SHORTS_MUSIC_VOLUME", "0.15")),
            x264_preset=_env("SHORTS_X264_PRESET", "medium"),
            uploader=_env("SHORTS_UPLOADER", "local").lower(),
            youtube_privacy=_privacy(_env("SHORTS_YOUTUBE_PRIVACY", "private")),
            youtube_signin_days=int(_env("SHORTS_YOUTUBE_SIGNIN_DAYS", "7")),
            notify_email=_env("SHORTS_NOTIFY_EMAIL"),
            output_dir=Path(_env("SHORTS_OUTPUT_DIR", "output")),
            state_dir=Path(_env("SHORTS_STATE_DIR", "state")),
        )

    def offline(self) -> "Config":
        """No network, no keys: sample news, template script, silent voice, local upload."""
        return replace(self, sources=["sample"], llm_provider="none", voice="silent", voice_lineup=[],
                       tavily_api_key="", uploader="local", notify_email="", web=False, shadow=False)
