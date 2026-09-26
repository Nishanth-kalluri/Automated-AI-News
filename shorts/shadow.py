"""The shadow week: how the web research path (phase 2) compares with the live path, day by day.

``pipeline.run_shadow`` runs the web path on the same candidates after the live episode is
published. This module holds the comparison: which picks differ, how widely each side's picks
are covered, what the researchers found, how the writer did, one fact check that judges both
scripts on the same material, and the cost. ``state/shadow.jsonl`` keeps a row per day.
"""
from __future__ import annotations

import calendar
import json
import logging
from dataclasses import replace
from datetime import date
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING

from .checks import _material, norm_url, predicted_seconds, same_event, unsupported_numbers
from .config import Config
from .llm import LLM
from .models import Episode, Segment, Story

if TYPE_CHECKING:
    from .coverage import Coverage
    from .pipeline import RunRecord

log = logging.getLogger(__name__)
KEEP_ROWS = 31
TYPICAL_RUN_USD = 0.30  # what a live run is assumed to cost when budgeting the rest of the month


def budget_skip(month_spent_usd: float, live_usd: float, cfg: Config, today: date | None = None) -> str:
    """"budget" when a shadow run could leave too little for the live runs left this month; "" otherwise."""
    today = today or date.today()
    days_after_today = calendar.monthrange(today.year, today.month)[1] - today.day
    need = month_spent_usd + cfg.budget_usd + days_after_today * max(live_usd, TYPICAL_RUN_USD)
    return "budget" if need > cfg.monthly_budget_usd else ""


def _name(s: Story) -> str:
    return s.headline or s.title


def pair_picks(live: list[Story], shadow: list[Story]) -> tuple[list[tuple[Story, Story]], list[Story], list[Story]]:
    """(pairs of the same story, live only, shadow only). Same story: same link or same event."""
    pairs, used = [], set()
    for a in live:
        for j, b in enumerate(shadow):
            if j in used:
                continue
            if (norm_url(a.url) and norm_url(a.url) == norm_url(b.url)) or same_event(_name(a), _name(b)):
                pairs.append((a, b))
                used.add(j)
                break
    paired = [a for a, _ in pairs]
    return (pairs, [a for a in live if not any(a is p for p in paired)],
            [b for j, b in enumerate(shadow) if j not in used])


def _coverage(coverage: Coverage | None, stories: list[Story]) -> dict:
    if coverage is None or not stories:
        return {"hn_points": None, "news_outlets": None, "unknown": len(stories)}
    try:
        rows = list(coverage.lookup_many([_name(s) for s in stories], timeout=45).values())
    except Exception as exc:
        log.warning("shadow: coverage lookup failed (%s)", exc)
        return {"hn_points": None, "news_outlets": None, "unknown": len(stories)}
    pts = [r["hn_points"] for r in rows if r.get("hn_points") is not None]
    outlets = [r["news_outlets"] for r in rows if r.get("news_outlets") is not None]
    return {"hn_points": round(mean(pts), 1) if pts else None,
            "news_outlets": round(mean(outlets), 1) if outlets else None,
            "unknown": sum(1 for r in rows if r.get("hn_points") is None and r.get("news_outlets") is None)}


def _research(rec: RunRecord) -> dict:
    rows = rec.research
    statuses = ("verified", "thin", "stale", "wrong_story", "failed")
    return {"status": {k: sum(1 for r in rows if (r.get("status") or "failed") == k) for k in statuses},
            "quotes_kept": sum(len(r.get("quotes") or []) for r in rows),
            "quotes_dropped": sum(int(r.get("dropped_quotes") or 0) for r in rows),
            "dated": sum(1 for r in rows if r.get("first_reported")),
            "researcher_errors": sum(1 for r in rows if r.get("error")),
            "seconds": max((float(r.get("seconds") or 0) for r in rows), default=0.0),
            "swaps": rec.swaps, "tavily_credits": rec.tavily_credits}


def _writer(rec: RunRecord) -> dict:
    rounds = (rec.review or {}).get("rounds") or []
    flags = [int(r.get("fatal", 0)) + int(r.get("critic", 0)) for r in rounds]
    return {"first_round_flags": flags[0] if flags else None, "last_round_flags": flags[-1] if flags else None,
            "rounds": len(rounds), "fallbacks": len((rec.review or {}).get("fallbacks") or []),
            "predicted_seconds": round(predicted_seconds(rec.episode), 1) if rec.episode else None,
            "qa_passed": rec.qa.passed if rec.qa else None, "seconds": rec.qa.duration if rec.qa else None,
            "error": rec.error}


def _segment_for(rec: RunRecord, story: Story) -> Segment | None:
    if rec.episode is None:
        return None
    for s, seg in zip(rec.episode.stories, rec.episode.story_segments):
        if s is story:
            return seg
    return None


def cross_check(live: RunRecord, shadow: RunRecord, pairs: list[tuple[Story, Story]], critic: LLM | None,
                show: str, host: str) -> dict | None:
    """Both scripts judged by one critic call on the same material: the shadow's checked quotes plus
    the article text either side had. None when no shared story has quotes or there is no critic."""
    from .writer import CriticWriter

    items = []
    for a, b in pairs:
        seg_a, seg_b = _segment_for(live, a), _segment_for(shadow, b)
        if b.evidence and seg_a and seg_b:
            quotes = "\n".join(f'- "{e.quote}"' for e in b.evidence)
            body = f"Verified quotes:\n{quotes}\n\n{(a.body or b.body)[:6000]}"
            items.append((replace(b, evidence=[], body=body, first_reported=""), seg_a, seg_b))
    if not items or critic is None:
        return None
    stories, segments = [], [Segment(kind="intro", text="")]
    for material, seg_a, seg_b in items:
        stories += [material, material]
        segments += [seg_a, seg_b]
    segments.append(Segment(kind="outro", text=""))
    episode = Episode(title="", description="", tags=[], segments=segments, stories=stories)
    result = {"stories": len(items), "live_unsupported": 0, "shadow_unsupported": 0, "live_numbers": 0,
              "shadow_numbers": 0, "claims": []}
    for k, (material, seg_a, seg_b) in enumerate(items):
        for side, seg in (("live", seg_a), ("shadow", seg_b)):
            bad = unsupported_numbers(f"{seg.text} {seg.key_fact} {seg.headline}", _material(material))
            result[f"{side}_numbers"] += len(bad)
    try:
        issues = CriticWriter(critic, critic, show, host).review(episode, stories)
    except Exception as exc:
        log.warning("shadow: the cross-check failed (%s)", exc)
        result["error"] = str(exc)[:300]
        return result
    for issue in issues:
        if issue.index is None:
            continue
        side = "live" if issue.index % 2 == 0 else "shadow"
        result[f"{side}_unsupported"] += 1
        result["claims"].append({"side": side, "story": _name(items[issue.index // 2][0]),
                                 "detail": issue.detail[:300]})
    return result


def _cost(rec: RunRecord) -> dict:
    usage = rec.usage
    return {"usd": round(usage.total_usd, 4) if usage else 0.0, "llm_calls": usage.calls if usage else 0,
            "tavily_credits": rec.tavily_credits}


def compare(live: RunRecord, shadow: RunRecord, coverage: Coverage | None, critic: LLM | None,
            show: str, host: str) -> dict:
    pairs, live_only, shadow_only = pair_picks(live.stories, shadow.stories)
    return {
        "date": date.today().isoformat(),
        "picks": {"shared": len(pairs), "live_only": [_name(s) for s in live_only],
                  "shadow_only": [_name(s) for s in shadow_only]},
        "live_stories": [{"headline": _name(s), "url": s.url} for s in live.stories],
        "shadow_stories": [{"headline": _name(s), "url": s.url, "checked": s.checked,
                            "first_reported": s.first_reported} for s in shadow.stories],
        "coverage": {"live": _coverage(coverage, live.stories), "shadow": _coverage(coverage, shadow.stories)},
        "research": _research(shadow),
        "writer": {"live": _writer(live), "shadow": _writer(shadow)},
        "cross_check": cross_check(live, shadow, pairs, critic, show, host),
        "cost": {"live": _cost(live), "shadow": _cost(shadow)},
    }


def row(report: dict) -> dict:
    """The day's line for state/shadow.jsonl: headlines, links and numbers only."""
    keep = ("date", "picks", "live_stories", "shadow_stories", "coverage", "research", "writer", "cost",
            "skipped", "error")
    out = {k: report[k] for k in keep if k in report}
    check = report.get("cross_check")
    if check is not None:
        out["cross_check"] = {k: v for k, v in check.items() if k != "claims"}
    return out


class ShadowLog:
    def __init__(self, path: Path):
        self.path = path

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    def append(self, entry: dict) -> int:
        """Adds a row, keeps the last 31, and returns how many rows there are."""
        rows = (self.rows() + [entry])[-KEEP_ROWS:]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
        tmp.replace(self.path)
        return len(rows)


def _cov(c: dict) -> str:
    outlets = "?" if c.get("news_outlets") is None else f"{c['news_outlets']:g}"
    pts = "?" if c.get("hn_points") is None else f"{c['hn_points']:g}"
    return f"{outlets} outlets / {pts} HN pts"


def _qa(w: dict) -> str:
    if w.get("error") and not w.get("qa_passed"):
        return f"failed ({w['error'][:120]})"
    if w.get("qa_passed") is None:
        return "no video"
    return f"{'passed' if w['qa_passed'] else 'failed'} {w.get('seconds') or 0:.0f} s"


def summary(report: dict, month_credits: int = 0, month_cap: int = 0) -> str:
    if report.get("skipped"):
        return f"The shadow run was skipped today ({report['skipped']}); the live episode is not affected."
    if report.get("error") and "picks" not in report:
        return f"The shadow run crashed ({report['error'][:300]}); the live episode is not affected."
    p, r, w, c = report["picks"], report["research"], report["writer"], report["cost"]
    st = r["status"]
    lines = [
        f"Picks: {p['shared']} shared; live only: {', '.join(p['live_only']) or 'none'}; "
        f"shadow only: {', '.join(p['shadow_only']) or 'none'}",
        f"Research: {st['verified']} verified, {st['thin']} thin, {st['stale']} stale, {st['wrong_story']} wrong "
        f"story, {st['failed']} failed; {r['quotes_kept']} quotes kept, {r['quotes_dropped']} dropped; "
        f"{r['swaps']} swaps",
        f"Coverage per pick: live {_cov(report['coverage']['live'])}, shadow {_cov(report['coverage']['shadow'])}",
    ]
    check = report.get("cross_check")
    if check is None:
        lines.append("Same fact-check on shared stories: skipped (no shared story with quotes)")
    else:
        lines.append(f"Same fact-check on {check['stories']} shared stories: live {check['live_unsupported']} "
                     f"unsupported, shadow {check['shadow_unsupported']}; numbers: live {check['live_numbers']}, "
                     f"shadow {check['shadow_numbers']}" + (" (critic failed)" if check.get("error") else ""))
    lines += [
        f"Writer: first-round flags live {w['live']['first_round_flags']} / shadow "
        f"{w['shadow']['first_round_flags']}; template fallbacks live {w['live']['fallbacks']} / shadow "
        f"{w['shadow']['fallbacks']}",
        f"QA: live {_qa(w['live'])}, shadow {_qa(w['shadow'])}",
        f"Cost: live ${c['live']['usd']:.2f}, shadow ${c['shadow']['usd']:.2f}; Tavily "
        f"{c['live']['tavily_credits']} + {c['shadow']['tavily_credits']} credits"
        + (f" (month {month_credits}/{month_cap})" if month_cap else ""),
        "The shadow video and its files are in the run's artifact, under shadow/.",
    ]
    return "\n".join(lines)
