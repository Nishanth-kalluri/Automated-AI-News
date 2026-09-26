from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

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
DEFAULT_LLM_MODELS = {"openai": "gpt-5", "anthropic": "claude-opus-5"}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


def _list(name: str, default: list[str]) -> list[str]:
    raw = _env(name)
    return [x.strip() for x in raw.split(",") if x.strip()] if raw else list(default)


@dataclass
class Config:
    sources: list[str]
    rss_feeds: list[str]
    stories_per_video: int
    max_age_hours: float
    # LLM that picks stories and writes the script: openai, anthropic, or none (templates).
    llm_provider: str
    llm_model: str
    openai_api_key: str
    anthropic_api_key: str
    # Newsletter inbox (AgentMail) and article reader (Tavily).
    agentmail_api_key: str
    agentmail_inbox: str
    tavily_api_key: str
    show_name: str
    host_name: str
    voice: str
    edge_voice: str
    edge_rate: str
    animator: str
    x264_preset: str
    uploader: str
    youtube_privacy: str
    notify_email: str
    output_dir: Path
    state_dir: Path

    @classmethod
    def from_env(cls) -> "Config":
        openai_key, anthropic_key = _env("OPENAI_API_KEY"), _env("ANTHROPIC_API_KEY")
        auto = "openai" if openai_key else "anthropic" if anthropic_key else "none"
        provider = _env("SHORTS_LLM_PROVIDER", auto)
        return cls(
            sources=_list("SHORTS_SOURCES", ["newsletter", "rss", "hackernews", "reddit"]),
            rss_feeds=_list("SHORTS_RSS_FEEDS", DEFAULT_RSS_FEEDS),
            stories_per_video=int(_env("SHORTS_STORIES_PER_VIDEO", "8")),
            max_age_hours=float(_env("SHORTS_MAX_AGE_HOURS", "30")),
            llm_provider=provider,
            llm_model=_env("SHORTS_LLM_MODEL", DEFAULT_LLM_MODELS.get(provider, "")),
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
            agentmail_api_key=_env("AGENTMAIL_API_KEY"),
            agentmail_inbox=_env("AGENTMAIL_INBOX"),
            tavily_api_key=_env("TAVILY_API_KEY"),
            show_name=_env("SHORTS_SHOW_NAME", "Duck Desk"),
            host_name=_env("SHORTS_HOST_NAME", "Quackers"),
            voice=_env("SHORTS_VOICE", "edge"),
            edge_voice=_env("SHORTS_EDGE_VOICE", "en-US-AnaNeural"),
            edge_rate=_env("SHORTS_EDGE_RATE", "+8%"),
            animator=_env("SHORTS_ANIMATOR", "puppet"),
            x264_preset=_env("SHORTS_X264_PRESET", "medium"),
            uploader=_env("SHORTS_UPLOADER", "local"),
            youtube_privacy=_env("SHORTS_YOUTUBE_PRIVACY", "private"),
            notify_email=_env("SHORTS_NOTIFY_EMAIL"),
            output_dir=Path(_env("SHORTS_OUTPUT_DIR", "output")),
            state_dir=Path(_env("SHORTS_STATE_DIR", "state")),
        )

    def offline(self) -> "Config":
        """No network, no keys: sample news, template script, silent voice, local upload."""
        return replace(self, sources=["sample"], llm_provider="none", voice="silent",
                       tavily_api_key="", uploader="local", notify_email="")
