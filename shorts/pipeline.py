"""Wires the stages together. Each stage is chosen by config and writes its output into the run folder.

    01-candidates.json  01-candidates.full.json  02-picks.json  03-episode.json  audio/  host/
    cards/  timeline.json  captions.ass  short.mp4  qa.json  upload.json  cost.json

With an LLM configured and SHORTS_AGENTS on (the default), the editor and writer run as agents
with check-and-repair loops; without one, every stage uses its no-key fallback.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, replace
from datetime import date, datetime
from pathlib import Path

from . import composer
from .character import build_animator
from .checks import TARGET_MAX_SECONDS
from .config import Config
from .llm import SpendLedger, Usage, build_llm, with_model
from .models import Story
from .notify import build_notifier, notify
from .qa import check
from .research import build_researcher, research
from .selection import SeenStore, build_editor, pick_with_fallback
from .sources import build_sources, fetch_all
from .upload import build_uploader
from .visuals import StoryCards
from .voice import build_voice, narrate
from .writer import build_writer, episode_json, shorten, write_episode

log = logging.getLogger(__name__)
VOICE_PASSES = 3  # voice, and up to two trims if it runs long


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
    ledger = SpendLedger(cfg.state_dir / "spend.json")
    usage = Usage(run_cap_usd=cfg.budget_usd, month_cap_usd=cfg.monthly_budget_usd,
                  month_spent_usd=ledger.this_month())
    try:
        return _run(cfg, run_dir, usage, upload)
    finally:  # record what was spent even when the run fails
        ledger.add(usage.total_usd)
        _dump(run_dir / "cost.json", {"usd": round(usage.total_usd, 4), "run_cap_usd": cfg.budget_usd,
                                      "month_usd": round(ledger.this_month(), 4),
                                      "month_cap_usd": cfg.monthly_budget_usd, "llm_calls": usage.calls})


def _run(cfg: Config, run_dir: Path, usage: Usage, upload: bool) -> Path:
    seen = SeenStore(cfg.state_dir / "seen_urls.json")
    llm = build_llm(cfg, usage)
    agents = cfg.agents and llm is not None
    offline = cfg.sources == ["sample"]
    n = cfg.stories_per_video

    log.info("[1/9] gathering news from %s", ", ".join(cfg.sources))
    candidates = fetch_all(build_sources(cfg))
    if not candidates:
        raise RuntimeError("No stories from any news source; see the warnings above.")
    _dump(run_dir / "01-candidates.json", _story_rows(candidates))
    _dump(run_dir / "01-candidates.full.json", _story_rows(candidates, with_body=True))  # to replay the editor

    editor = build_editor(with_model(llm, cfg.editor_model), cfg.max_age_hours, agents, cfg.max_repairs)
    log.info("[2/9] %s editor picking %d of %d candidates", editor.name, n, len(candidates))
    stories = pick_with_fallback(editor, candidates, n, seen, cfg.max_age_hours)
    if len(stories) < n:
        raise RuntimeError(f"Only {len(stories)} fresh, unused AI stories today (need {n}). "
                           "Try raising SHORTS_MAX_AGE_HOURS or lowering SHORTS_STORIES_PER_VIDEO.")
    for s in stories:
        log.info("      %s  (%s)", s.headline or s.title, ", ".join(s.outlets or [s.source]))

    researcher = build_researcher(cfg.tavily_api_key)
    log.info("[3/9] reading source articles with %s", researcher.name)
    research(researcher, stories)
    _dump(run_dir / "02-picks.json", _story_rows(stories, with_body=True))

    writer = build_writer(with_model(llm, cfg.writer_model), cfg.show_name, cfg.host_name, agents=agents,
                          critic=with_model(llm, cfg.checker_model), max_repairs=cfg.max_repairs)
    log.info("[4/9] writing the episode with %s writer", writer.name)
    episode = write_episode(writer, stories, cfg.show_name, cfg.host_name)
    (run_dir / "03-episode.json").write_text(episode_json(episode))

    voice = build_voice(cfg)
    log.info("[5/9] voicing %d segments with %s", len(episode.segments), voice.name)
    for attempt in range(1, VOICE_PASSES + 1):
        vo = narrate(voice, episode, run_dir / "audio")
        log.info("      %.1fs of audio", vo.duration)
        if vo.duration <= TARGET_MAX_SECONDS or attempt == VOICE_PASSES:
            break
        log.info("      longer than %ds; trimming the script", TARGET_MAX_SECONDS)
        episode = shorten(writer, episode, stories, vo.duration)
        (run_dir / "03-episode.json").write_text(episode_json(episode))
    if voice.name == "silent" and not offline:  # SHORTS_VOICE=silent, or edge-tts isn't installed
        vo.silent_segments = list(range(len(episode.segments)))

    animator = build_animator(cfg.animator, seed=date.today().toordinal())
    log.info("[6/9] animating the host with %s", animator.name)
    host = animator.animate(vo, run_dir / "host")

    log.info("[7/9] drawing story cards")
    cards = StoryCards(cfg.show_name)
    backgrounds = cards.render(episode, run_dir / "cards")
    desk = cards.desk(run_dir / "cards" / "desk.png")

    log.info("[8/9] rendering the video")
    video = composer.render(vo, backgrounds, desk, host, run_dir / "short.mp4", cfg.x264_preset)
    report = check(video, episode, n, vo, allow_sample=offline)
    _dump(run_dir / "qa.json", asdict(report))
    for w in report.warnings:
        log.warning("qa: %s", w)
    if not report.passed:
        raise RuntimeError("Quality check failed: " + "; ".join(report.problems))

    uploader = build_uploader(cfg) if upload else build_uploader(replace(cfg, uploader="local"))
    log.info("[9/9] publishing with %s uploader", uploader.name)
    result = uploader.upload(video, episode)
    log.info("      %s", result.location)

    if not offline:
        seen.add(stories)
    headlines = "\n".join(f"{i}. {s.headline}  {s.url}" for i, s in enumerate(episode.story_segments, 1))
    notify(build_notifier(cfg),
           f"New episode ready: {episode.title}",
           f"{result.location}\n\nUploaded as {cfg.youtube_privacy if result.uploader == 'youtube' else 'a local file'}. "
           f"Review it and make it public in YouTube Studio.\n\nLength: {report.duration:.0f}s\n\n{headlines}\n\n"
           f"Cost: ${usage.total_usd:.2f} (${usage.month_spent_usd + usage.total_usd:.2f} this month)\n"
           f"Warnings: {'; '.join(report.warnings) or 'none'}")
    log.info("done: %s", video)
    return video
