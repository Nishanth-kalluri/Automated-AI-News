"""Wires the stages together. Each stage is chosen by config and writes its output into the run folder.

    01-candidates.json  01-candidates.full.json  02-picks.json  02-research.json  02-coverage.json
    03-episode.json  03-review.json  audio/  host/  cards/  timeline.json  captions.ass  short.mp4
    qa.json  upload.json  cost.json  shadow/

With an LLM configured and SHORTS_AGENTS on (the default), the editor and writer run as agents
with check-and-repair loops; without one, every stage uses its no-key fallback. SHORTS_WEB=on
adds the web tools: coverage for the editor and one research agent per story. SHORTS_SHADOW=on
(with SHORTS_WEB off) runs that web path a second time after the episode is published, on the
same news, into shadow/, and emails how the two compare; it never uploads or changes what aired.
"""
from __future__ import annotations

import copy
import json
import logging
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime
from pathlib import Path

from . import composer
from .character import build_animator
from .checks import TARGET_MAX_SECONDS
from .config import Config
from .coverage import Coverage
from .llm import SpendLedger, Usage, build_llm, with_model
from .models import Episode, Story
from .notify import build_notifier, notify
from .qa import QAReport, check
from .research import AgentResearcher, build_researcher, research, swap_failing
from .selection import HeuristicEditor, SeenStore, build_editor, pick_with_fallback, settle
from .sources import build_sources, fetch_all
from .upload import build_uploader
from .visuals import StoryCards
from .voice import build_voice, narrate
from .web import TavilyCredits, build_web
from .writer import build_writer, episode_json, shorten, write_episode

log = logging.getLogger(__name__)
VOICE_PASSES = 3  # voice, and up to two trims if it runs long


@dataclass
class RunRecord:
    """What one run saw and made. The shadow run starts from production's candidates and
    aired list, and the comparison reads both records."""

    usage: Usage | None = None
    candidates: list[Story] | None = None  # as fetched, before any stage touched them
    seen: SeenStore | None = None  # as it was before this run
    stories: list[Story] = field(default_factory=list)
    episode: Episode | None = None
    qa: QAReport | None = None
    review: dict = field(default_factory=dict)
    research: list[dict] = field(default_factory=list)
    swaps: int = 0
    coverage: Coverage | None = None
    tavily_credits: int = 0
    error: str = ""


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


def _cost(usage: Usage, ledger: SpendLedger, cfg: Config, credits: TavilyCredits, tavily_run: int) -> dict:
    return {"usd": round(usage.total_usd, 4), "run_cap_usd": cfg.budget_usd,
            "month_usd": round(ledger.this_month(), 4), "month_cap_usd": cfg.monthly_budget_usd,
            "llm_calls": usage.calls, "tavily_credits_run": tavily_run,
            "tavily_credits_month": credits.this_month(), "tavily_month_cap": cfg.tavily_monthly_credits}


def run(cfg: Config, *, upload: bool = False) -> Path:
    run_dir = cfg.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger = SpendLedger(cfg.state_dir / "spend.json")
    usage = Usage(run_cap_usd=cfg.budget_usd, month_cap_usd=cfg.monthly_budget_usd,
                  month_spent_usd=ledger.this_month(), on_add=ledger.add)  # saved call by call
    credits = TavilyCredits(cfg.state_dir / "tavily.json", cfg.tavily_monthly_credits, cfg.tavily_run_credits)
    live = RunRecord(usage=usage)
    finished = False  # False only when interrupted (Ctrl-C, a cancelled job): then no shadow
    try:
        video = _run(cfg, run_dir, usage, upload, credits, live)
        finished = True
        return video
    except Exception as exc:
        live.error = f"{type(exc).__name__}: {exc}"[:500]
        finished = True
        raise
    finally:
        live.tavily_credits = credits.run_used
        _dump(run_dir / "cost.json", _cost(usage, ledger, cfg, credits, credits.run_used))
        if finished and cfg.shadow and not cfg.web and live.candidates and cfg.sources != ["sample"]:
            run_shadow(cfg, run_dir / "shadow", ledger, credits, live)


def run_shadow(cfg: Config, out_dir: Path, ledger: SpendLedger, credits: TavilyCredits, live: RunRecord) -> None:
    """The web path on the live run's candidates, into ``out_dir``, then the comparison email.

    Runs after the live episode is published and never raises, so it can't change or stop what airs.
    """
    from . import shadow

    notifier = build_notifier(cfg)
    subject = f"{cfg.show_name} shadow run: web research vs live"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        days = shadow.ShadowLog(cfg.state_dir / "shadow.jsonl")
        reason = shadow.budget_skip(ledger.this_month(), live.usage.total_usd if live.usage else 0.0, cfg)
        if reason:
            log.warning("shadow: skipped (%s)", reason)
            report = {"date": date.today().isoformat(), "skipped": reason}
            _dump(out_dir / "compare.json", report)
            days.append(report)
            notify(notifier, subject, shadow.summary(report))
            return
        log.info("shadow: running the web research path on the same news")
        scfg = replace(cfg, web=True, shadow=False, uploader="local", x264_preset="ultrafast")
        usage = Usage(run_cap_usd=cfg.budget_usd, month_cap_usd=cfg.monthly_budget_usd,
                      month_spent_usd=ledger.this_month(), on_add=ledger.add)
        credits.new_run()
        rec = RunRecord(candidates=live.candidates, seen=live.seen, usage=usage)
        try:
            _run(scfg, out_dir, usage, False, credits, rec, shadow=True)
        except Exception as exc:
            rec.error = f"{type(exc).__name__}: {exc}"[:500]
            log.warning("shadow: the run failed (%s)", rec.error)
        finally:
            rec.tavily_credits = credits.run_used
            _dump(out_dir / "cost.json", _cost(usage, ledger, cfg, credits, credits.run_used))
        critic = with_model(build_llm(scfg, usage), cfg.checker_model)
        report = shadow.compare(live, rec, rec.coverage or Coverage(cfg.max_age_hours), critic,
                                cfg.show_name, cfg.host_name)
        _dump(out_dir / "compare.json", report)
        day = days.append(shadow.row(report))
        notify(notifier, f"{cfg.show_name} shadow day {day}: web research vs live",
               shadow.summary(report, credits.this_month(), cfg.tavily_monthly_credits))
    except Exception as exc:  # the live episode is already out; only report it
        log.warning("shadow: failed (%s)", exc)
        report = {"date": date.today().isoformat(), "error": f"{type(exc).__name__}: {exc}"[:500]}
        try:
            _dump(out_dir / "compare.json", report)
        except OSError:
            pass
        notify(notifier, subject, shadow.summary(report))


def _run(cfg: Config, run_dir: Path, usage: Usage, upload: bool, credits: TavilyCredits, rec: RunRecord,
         *, shadow: bool = False) -> Path:
    if rec.seen is not None:
        seen = copy.deepcopy(rec.seen)
    else:
        seen = SeenStore(cfg.state_dir / "seen_urls.json")
        rec.seen = copy.deepcopy(seen)
    llm = build_llm(cfg, usage)
    agents = cfg.agents and llm is not None
    offline = cfg.sources == ["sample"]
    n = cfg.stories_per_video
    web = build_web(cfg, credits)
    rec.coverage = web.coverage if web else None

    if rec.candidates is not None:
        candidates = copy.deepcopy(rec.candidates)
        log.info("[1/9] using the same %d candidates as the live run", len(candidates))
    else:
        log.info("[1/9] gathering news from %s", ", ".join(cfg.sources))
        candidates = fetch_all(build_sources(cfg))
        if not candidates:
            raise RuntimeError("No stories from any news source; see the warnings above.")
        rec.candidates = copy.deepcopy(candidates)
        _dump(run_dir / "01-candidates.json", _story_rows(candidates))
        _dump(run_dir / "01-candidates.full.json", _story_rows(candidates, with_body=True))  # to replay the editor

    editor = build_editor(with_model(llm, cfg.editor_model), cfg.max_age_hours, agents, cfg.max_repairs,
                          coverage=web.coverage if web else None, tavily=web.tavily if web else None)
    log.info("[2/9] %s editor picking %d of %d candidates", editor.name, n, len(candidates))
    stories = pick_with_fallback(editor, candidates, n, seen, cfg.max_age_hours)
    if len(stories) < n:
        raise RuntimeError(f"Only {len(stories)} fresh, unused AI stories today (need {n}). "
                           "Try raising SHORTS_MAX_AGE_HOURS or lowering SHORTS_STORIES_PER_VIDEO.")
    for s in stories:
        log.info("      %s  (%s)", s.headline or s.title, ", ".join(s.outlets or [s.source]))

    researcher = build_researcher(cfg, with_model(llm, cfg.checker_model), web, candidates, credits, seen.urls)
    log.info("[3/9] reading source articles with %s", researcher.name)
    research(researcher, stories)
    if isinstance(researcher, AgentResearcher):
        picked = list(stories)
        pool = [s for s in HeuristicEditor(cfg.max_age_hours).pick(candidates, len(candidates), seen)
                if not any(s is p for p in picked)]
        stories = swap_failing(researcher, stories, list(getattr(editor, "alternates", [])) + pool,
                               lambda picks: settle(picks, candidates, n, seen, cfg.max_age_hours))
        rec.swaps = sum(1 for s in stories if not any(s is p for p in picked))
        rec.research = researcher.rows
        _dump(run_dir / "02-research.json", researcher.rows)
    rec.stories = stories
    _dump(run_dir / "02-picks.json", _story_rows(stories, with_body=True))
    if web:
        _dump(run_dir / "02-coverage.json", web.coverage.rows())

    writer = build_writer(with_model(llm, cfg.writer_model), cfg.show_name, cfg.host_name, agents=agents,
                          critic=with_model(llm, cfg.checker_model), max_repairs=cfg.max_repairs)
    log.info("[4/9] writing the episode with %s writer", writer.name)
    episode = rec.episode = write_episode(writer, stories, cfg.show_name, cfg.host_name)
    (run_dir / "03-episode.json").write_text(episode_json(episode))
    rec.review = getattr(writer, "report", {})
    _dump(run_dir / "03-review.json", rec.review)

    voice = build_voice(cfg)
    log.info("[5/9] voicing %d segments with %s", len(episode.segments), voice.name)
    for attempt in range(1, VOICE_PASSES + 1):
        vo = narrate(voice, episode, run_dir / "audio")
        log.info("      %.1fs of audio", vo.duration)
        if vo.duration <= TARGET_MAX_SECONDS or attempt == VOICE_PASSES:
            break
        log.info("      longer than %ds; trimming the script", TARGET_MAX_SECONDS)
        episode = rec.episode = shorten(writer, episode, stories, vo.duration)
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
    report = rec.qa = check(video, episode, n, vo, allow_sample=offline)
    _dump(run_dir / "qa.json", asdict(report))
    for w in report.warnings:
        log.warning("qa: %s", w)
    if not report.passed:
        raise RuntimeError("Quality check failed: " + "; ".join(report.problems))

    live_upload = upload and not shadow
    uploader = build_uploader(cfg) if live_upload else build_uploader(replace(cfg, uploader="local"))
    log.info("[9/9] publishing with %s uploader", uploader.name)
    result = uploader.upload(video, episode)
    log.info("      %s", result.location)
    if shadow:
        return video

    if not offline:
        seen.add(stories)
    headlines = "\n".join(f"{i}. {s.headline}  {s.url}" for i, s in enumerate(episode.story_segments, 1))
    notify(build_notifier(cfg),
           f"New episode ready: {episode.title}",
           f"{result.location}\n\nUploaded as {cfg.youtube_privacy if result.uploader == 'youtube' else 'a local file'}. "
           f"Review it and make it public in YouTube Studio.\n\nLength: {report.duration:.0f}s\n\n{headlines}\n\n"
           f"Cost: ${usage.total_usd:.2f} (${usage.month_spent_usd + usage.total_usd:.2f} this month); "
           f"Tavily {credits.run_used} credits ({credits.this_month()}/{cfg.tavily_monthly_credits} this month)\n"
           f"Warnings: {'; '.join(report.warnings) or 'none'}")
    log.info("done: %s", video)
    return video
