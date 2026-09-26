"""Wires the stages together. Each stage is chosen by config and writes its output into the run folder.

    01-candidates.json  02-picks.json  03-episode.json  audio/  host/  cards/
    timeline.json  captions.ass  short.mp4  qa.json  upload.json  cost.json
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

from . import composer
from .character import build_animator
from .config import Config
from .llm import Usage, build_llm
from .models import Story
from .notify import build_notifier, notify
from .qa import check
from .research import build_researcher, research
from .selection import SeenStore, build_editor, pick_with_fallback
from .sources import build_sources, fetch_all
from .upload import build_uploader
from .visuals import StoryCards
from .voice import build_voice, narrate
from .writer import build_writer, episode_json, write_episode

log = logging.getLogger(__name__)


def _dump(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))


def _story_rows(stories: list[Story], with_body: bool = False) -> list[dict]:
    rows = []
    for s in stories:
        row = asdict(s)
        if not with_body:
            row["body"] = f"{len(s.body)} chars" if s.body else ""
        rows.append(row)
    return rows


def run(cfg: Config, *, upload: bool = False) -> Path:
    run_dir = cfg.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    seen = SeenStore(cfg.state_dir / "seen_urls.json")
    usage = Usage()
    llm = build_llm(cfg, usage)

    log.info("[1/9] gathering news from %s", ", ".join(cfg.sources))
    candidates = fetch_all(build_sources(cfg))
    _dump(run_dir / "01-candidates.json", _story_rows(candidates))

    editor = build_editor(llm, cfg.max_age_hours)
    log.info("[2/9] %s editor picking %d of %d candidates", editor.name, cfg.stories_per_video, len(candidates))
    stories = pick_with_fallback(editor, candidates, cfg.stories_per_video, seen, cfg.max_age_hours)
    if not stories:
        raise RuntimeError("No fresh, unused AI stories found. Try raising SHORTS_MAX_AGE_HOURS.")
    for s in stories:
        log.info("      %s  (%s)", s.headline or s.title, ", ".join(s.outlets or [s.source]))

    researcher = build_researcher(cfg.tavily_api_key)
    log.info("[3/9] reading source articles with %s", researcher.name)
    research(researcher, stories)
    _dump(run_dir / "02-picks.json", _story_rows(stories, with_body=True))

    writer = build_writer(llm, cfg.show_name, cfg.host_name)
    log.info("[4/9] writing the episode with %s writer", writer.name)
    episode = write_episode(writer, stories, cfg.show_name, cfg.host_name)
    (run_dir / "03-episode.json").write_text(episode_json(episode))

    voice = build_voice(cfg)
    log.info("[5/9] voicing %d segments with %s", len(episode.segments), voice.name)
    vo = narrate(voice, episode, run_dir / "audio")
    log.info("      %.1fs of audio", vo.duration)

    animator = build_animator(cfg.animator, seed=date.today().toordinal())
    log.info("[6/9] animating the host with %s", animator.name)
    host = animator.animate(vo, run_dir / "host")

    log.info("[7/9] drawing story cards")
    cards = StoryCards(cfg.show_name)
    backgrounds = cards.render(episode, run_dir / "cards")
    desk = cards.desk(run_dir / "cards" / "desk.png")

    log.info("[8/9] rendering the video")
    video = composer.render(vo, backgrounds, desk, host, run_dir / "short.mp4", cfg.x264_preset)
    report = check(video, episode, cfg.stories_per_video)
    _dump(run_dir / "qa.json", asdict(report))
    _dump(run_dir / "cost.json", {"llm_calls": usage.calls, "tavily_articles": sum(bool(s.body) for s in stories
                                                                                    if s.url)})
    for w in report.warnings:
        log.warning("qa: %s", w)
    if not report.passed:
        raise RuntimeError("Quality check failed: " + "; ".join(report.problems))

    uploader = build_uploader(cfg) if upload else build_uploader(_local(cfg))
    log.info("[9/9] publishing with %s uploader", uploader.name)
    result = uploader.upload(video, episode)
    log.info("      %s", result.location)

    seen.add([s for s in stories if not s.url.startswith("https://example.com/sample/")])
    headlines = "\n".join(f"{i}. {s.headline}  {s.url}" for i, s in enumerate(episode.story_segments, 1))
    tokens = sum(c["input_tokens"] + c["output_tokens"] for c in usage.calls)
    notify(build_notifier(cfg),
           f"New episode ready: {episode.title}",
           f"{result.location}\n\nUploaded as {cfg.youtube_privacy if result.uploader == 'youtube' else 'a local file'}. "
           f"Review it and make it public in YouTube Studio.\n\nLength: {report.duration:.0f}s\n\n{headlines}\n\n"
           f"LLM tokens used: {tokens}\nWarnings: {'; '.join(report.warnings) or 'none'}")
    log.info("done: %s", video)
    return video


def _local(cfg: Config) -> Config:
    from dataclasses import replace

    return replace(cfg, uploader="local")
