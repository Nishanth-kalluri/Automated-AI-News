import json
from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import pytest
import requests

from shorts import pipeline, qa, shadow
from shorts.config import Config
from shorts.llm import Usage
from shorts.media import media_duration
from shorts.models import Episode, Evidence, Segment, UploadResult
from shorts.pipeline import RunRecord
from shorts.qa import QAReport
from shorts.shadow import KEEP_ROWS, ShadowLog, budget_skip, compare, pair_picks, row, summary
from shorts.voice import SilentVoice
from tests.test_agents import ScriptedLLM, _pick, _story

QUOTE = "GPT-6 Sol is 40 percent faster than GPT-5"
ARTICLE = "OpenAI released GPT-6 Sol to all ChatGPT users today. BODYMARKER"


# --- compare() and the day's row -------------------------------------------------------------

def _named(title, url=None, **kw):
    s = _story(title, url=url, **kw)
    s.headline = title
    return s


def _record(stories, texts, **kw):
    segments = ([Segment("intro", "Hello pond.")] + [Segment("story", t, s.headline) for s, t in zip(stories, texts)]
                + [Segment("outro", "Bye.")])
    return RunRecord(stories=stories, episode=Episode("t", "d", [], segments, stories=stories), **kw)


class FakeCoverage:
    def __init__(self, rows=None, error=None):
        self.rows, self.error, self.asked = rows or {}, error, []

    def lookup_many(self, headlines, timeout=30):
        self.asked.append(list(headlines))
        if self.error:
            raise self.error
        return {h: {"headline": h, **self.rows.get(h, {"hn_points": None, "news_outlets": None})} for h in headlines}


def _day():
    a = _named("OpenAI launches GPT-6 Sol", url="https://openai.com/gpt-6?utm_source=feed")
    b = _named("Meta releases Llama 5", url="https://meta.com/llama-5")
    c = _named("EU passes AI audit rules", url="https://eu.example/rules")
    a2 = _named("ChatGPT gets a new default model", url="https://www.openai.com/gpt-6/", body=ARTICLE)
    a2.evidence, a2.first_reported, a2.checked = [Evidence(QUOTE, a2.url)], "2026-09-25", "verified"
    b2 = _named("Meta's Llama 5 is out", url="https://news.example/llama")
    b2.evidence, b2.checked = [Evidence("Llama 5 ships with open weights", b2.url)], "thin"
    d = _named("Nvidia unveils Rubin chip", url="https://nvidia.com/rubin")
    live_usage, shadow_usage = Usage(), Usage()
    live_usage.add("writer", "gpt-5", 200_000, 0)
    shadow_usage.add("writer", "gpt-5", 400_000, 0)
    shadow_usage.add("research", "gpt-5", 0, 50_000)
    live = _record([a, b, c], ["Sol is out, and it is 900 percent better.", "Meta shipped Llama 5 today.",
                               "Europe wants audits of AI."],
                   usage=live_usage, qa=QAReport(True, 120.0),
                   review={"rounds": [{"fatal": 1, "critic": 2}, {"fatal": 0, "critic": 0}],
                           "fallbacks": [{"part": "title"}]})
    research_rows = [
        {"status": "verified", "quotes": [QUOTE, "x"], "dropped_quotes": 1, "first_reported": "2026-09-25",
         "seconds": 12.5, "error": ""},
        {"status": "thin", "quotes": [], "dropped_quotes": 0, "first_reported": "", "seconds": 30, "error": ""},
        {"status": "failed", "quotes": [], "dropped_quotes": 0, "seconds": 90, "error": "timed out"},
        {"headline": "not investigated", "status": ""},
        {"status": "stale", "first_reported": "2026-09-01"},
        {"status": "wrong_story"},
    ]
    shadow_rec = _record([b2, a2, d], ["Meta's Llama 5 is out with 3 new features.",
                                        "ChatGPT's new model is 40 percent faster.", "Nvidia has a chip."],
                         usage=shadow_usage, research=research_rows, swaps=1, tavily_credits=9,
                         error="RuntimeError: render failed")
    return live, shadow_rec


LIVE_COVERAGE = {"OpenAI launches GPT-6 Sol": {"hn_points": 10, "news_outlets": 3},
                 "Meta releases Llama 5": {"hn_points": None, "news_outlets": 5},
                 "EU passes AI audit rules": {"hn_points": 20, "news_outlets": None},
                 "ChatGPT gets a new default model": {"hn_points": 7, "news_outlets": 2}}
CRITIC_REPLY = {"intro": [], "outro": [], "segments": [
    {"story": 1, "unsupported": ["900 percent better"]},
    {"story": 2, "unsupported": [f'the quote "{QUOTE}" says otherwise']},
    {"story": 4, "unsupported": ["3 new features"]}]}


def test_pair_picks_matches_by_link_or_event_and_lists_the_rest():
    live, shadow_rec = _day()
    x, y = _named("Some story without a link", url=""), _named("Apple adds AI to Siri", url="")
    a, b, c = live.stories
    b2, a2, d = shadow_rec.stories
    pairs, live_only, shadow_only = pair_picks([a, b, c, x], [b2, a2, d, y])
    assert pairs[0][0] is a and pairs[0][1] is a2  # same link after normalising
    assert pairs[1][0] is b and pairs[1][1] is b2  # different links, same event
    assert len(pairs) == 2
    assert live_only == [c, x] and shadow_only == [d, y]  # two missing links are not the same link
    pairs, live_only, shadow_only = pair_picks([a, _named("OpenAI launches GPT-6 Sol", url="https://o.ai/x")], [a2])
    assert len(pairs) == 1 and len(live_only) == 1 and shadow_only == []  # a shadow pick pairs once


def test_compare_counts_coverage_research_and_cost():
    live, shadow_rec = _day()
    cov = FakeCoverage(LIVE_COVERAGE)
    report = compare(live, shadow_rec, cov, None, "Duck Desk", "Quackers")
    assert report["date"] == date.today().isoformat()
    assert report["picks"] == {"shared": 2, "live_only": ["EU passes AI audit rules"],
                               "shadow_only": ["Nvidia unveils Rubin chip"]}
    assert report["live_stories"][0] == {"headline": "OpenAI launches GPT-6 Sol",
                                         "url": "https://openai.com/gpt-6?utm_source=feed"}
    assert report["shadow_stories"][1] == {"headline": "ChatGPT gets a new default model",
                                           "url": "https://www.openai.com/gpt-6/", "checked": "verified",
                                           "first_reported": "2026-09-25"}
    # unknown counts are left out of the means, not counted as 0
    assert report["coverage"]["live"] == {"hn_points": 15.0, "news_outlets": 4.0, "unknown": 0}
    assert report["coverage"]["shadow"] == {"hn_points": 7.0, "news_outlets": 2.0, "unknown": 2}
    assert report["research"] == {
        "status": {"verified": 1, "thin": 1, "stale": 1, "wrong_story": 1, "failed": 2},
        "quotes_kept": 2, "quotes_dropped": 1, "dated": 2, "researcher_errors": 1, "seconds": 90.0,
        "swaps": 1, "tavily_credits": 9}
    w = report["writer"]
    assert (w["live"]["first_round_flags"], w["live"]["last_round_flags"], w["live"]["rounds"],
            w["live"]["fallbacks"], w["live"]["qa_passed"], w["live"]["seconds"]) == (3, 0, 2, 1, True, 120.0)
    assert (w["shadow"]["first_round_flags"], w["shadow"]["rounds"], w["shadow"]["qa_passed"],
            w["shadow"]["error"]) == (None, 0, None, "RuntimeError: render failed")
    assert w["live"]["predicted_seconds"] > 0
    assert report["cost"] == {"live": {"usd": 0.25, "llm_calls": 1, "tavily_credits": 0},
                              "shadow": {"usd": 1.0, "llm_calls": 2, "tavily_credits": 9}}
    assert report["cross_check"] is None  # no critic


def test_coverage_is_unknown_without_backends_or_when_they_fail():
    live, shadow_rec = _day()
    report = compare(live, shadow_rec, None, None, "S", "H")
    assert report["coverage"]["live"] == {"hn_points": None, "news_outlets": None, "unknown": 3}
    report = compare(live, shadow_rec, FakeCoverage(error=TimeoutError("slow")), None, "S", "H")
    assert report["coverage"]["shadow"] == {"hn_points": None, "news_outlets": None, "unknown": 3}
    report = compare(live, shadow_rec, FakeCoverage({}), None, "S", "H")
    assert report["coverage"]["live"] == {"hn_points": None, "news_outlets": None, "unknown": 3}


def test_cross_check_is_one_critic_call_split_into_live_and_shadow():
    live, shadow_rec = _day()
    critic = ScriptedLLM(CRITIC_REPLY)
    report = compare(live, shadow_rec, FakeCoverage(LIVE_COVERAGE), critic, "Duck Desk", "Quackers")
    assert [stage for stage, _ in critic.calls] == ["critic"]
    prompt = critic.calls[0][1]
    texts = ["Sol is out, and it is 900 percent better.", "ChatGPT's new model is 40 percent faster.",
             "Meta shipped Llama 5 today.", "Meta's Llama 5 is out with 3 new features."]
    positions = [prompt.index(t) for t in texts]
    assert positions == sorted(positions)  # live, shadow, live, shadow
    assert "Europe wants audits" not in prompt and "Nvidia has a chip" not in prompt  # only shared stories
    assert "Verified quotes" in prompt and QUOTE in prompt and "BODYMARKER" in prompt
    assert "First reported" not in prompt  # both sides judged on the same material
    check = report["cross_check"]
    assert (check["stories"], check["live_unsupported"], check["shadow_unsupported"]) == (2, 1, 2)
    assert (check["live_numbers"], check["shadow_numbers"]) == (1, 1)  # "900" and "3" are in no material
    assert [(c["side"], c["story"]) for c in check["claims"]] == [
        ("live", "ChatGPT gets a new default model"), ("shadow", "ChatGPT gets a new default model"),
        ("shadow", "Meta's Llama 5 is out")]
    assert "900 percent better" in check["claims"][0]["detail"]


def test_cross_check_material_holds_what_both_writers_were_given():
    live, shadow_rec = _day()
    a = live.stories[0]
    a.summary, a.key_fact, a.body = "Sol has a 2 million token context.", "2 million tokens", "LIVEBODY"
    live.episode.segments[1].text = "Sol reads 2 million tokens at once."
    critic = ScriptedLLM({"intro": [], "outro": [], "segments": []})
    check = compare(live, shadow_rec, None, critic, "Duck Desk", "Quackers")["cross_check"]
    prompt = critic.calls[0][1]
    assert "Sol has a 2 million token context." in prompt and "LIVEBODY" in prompt and "BODYMARKER" in prompt
    assert "Also headlined: OpenAI launches GPT-6 Sol" in prompt
    assert check["live_numbers"] == 0  # its own summary backs the number


def test_no_shared_evidence_means_no_cross_check_call():
    live, shadow_rec = _day()
    for s in shadow_rec.stories:
        s.evidence = []
    critic = ScriptedLLM()
    report = compare(live, shadow_rec, None, critic, "S", "H")
    assert report["cross_check"] is None and critic.calls == []
    live, shadow_rec = _day()
    shadow_rec.episode = None  # the shadow never got to a script
    report = compare(live, shadow_rec, None, critic, "S", "H")
    assert report["cross_check"] is None and critic.calls == []


def test_a_failing_critic_is_reported_not_raised():
    live, shadow_rec = _day()
    critic = ScriptedLLM(TimeoutError("critic timed out"))
    report = compare(live, shadow_rec, None, critic, "S", "H")
    check = report["cross_check"]
    assert "critic timed out" in check["error"] and check["live_unsupported"] == check["shadow_unsupported"] == 0
    assert check["live_numbers"] == 1
    assert "(critic failed)" in summary(report)


def test_row_holds_no_article_text_or_quotes():
    live, shadow_rec = _day()
    report = compare(live, shadow_rec, FakeCoverage(LIVE_COVERAGE), ScriptedLLM(CRITIC_REPLY), "S", "H")
    assert report["cross_check"]["claims"]  # the full report keeps the critic's findings
    line = row(report)
    text = json.dumps(line)
    assert QUOTE not in text and "BODYMARKER" not in text and "900 percent better" not in text
    assert "claims" not in line["cross_check"] and line["cross_check"]["shadow_unsupported"] == 2
    assert set(line) == {"date", "picks", "live_stories", "shadow_stories", "coverage", "research", "writer",
                         "cost", "cross_check"}
    assert row({"date": "2026-09-26", "skipped": "budget"}) == {"date": "2026-09-26", "skipped": "budget"}


def test_shadow_log_keeps_the_last_31_rows(tmp_path):
    days = ShadowLog(tmp_path / "state" / "shadow.jsonl")
    assert days.rows() == []
    counts = [days.append({"day": i}) for i in range(35)]
    assert counts[:3] == [1, 2, 3] and counts[-1] == KEEP_ROWS == 31
    assert [r["day"] for r in days.rows()] == list(range(4, 35))
    with days.path.open("a") as f:
        f.write("not json\n")
    assert days.append({"day": 35}) == 31 and days.rows()[-1] == {"day": 35}
    assert [p.name for p in days.path.parent.iterdir()] == ["shadow.jsonl"]


def test_budget_skip_leaves_room_for_the_rest_of_the_month():
    cfg = replace(Config.from_env(), budget_usd=0.60, monthly_budget_usd=18.0)
    sept_26 = date(2026, 9, 26)  # 4 more live runs this month, at least $0.30 each
    assert budget_skip(16.0, 0.10, cfg, today=sept_26) == ""
    assert budget_skip(16.3, 0.10, cfg, today=sept_26) == "budget"
    assert budget_skip(15.5, 0.50, cfg, today=sept_26) == "budget"  # today's live cost sets the pace
    assert budget_skip(17.3, 0.50, cfg, today=date(2026, 9, 30)) == ""  # the month's last run
    assert budget_skip(17.5, 0.50, cfg, today=date(2026, 9, 30)) == "budget"
    assert budget_skip(9.2, 0.10, cfg, today=date(2026, 2, 1)) == ""
    assert budget_skip(9.4, 0.10, cfg, today=date(2026, 2, 1)) == "budget"


def test_summary_reads_for_a_normal_skipped_and_crashed_day():
    live, shadow_rec = _day()
    report = compare(live, shadow_rec, FakeCoverage(LIVE_COVERAGE), ScriptedLLM(CRITIC_REPLY), "S", "H")
    text = summary(report, month_credits=12, month_cap=700)
    for part in ("Picks: 2 shared; live only: EU passes AI audit rules; shadow only: Nvidia unveils Rubin chip",
                 "Research: 1 verified, 1 thin, 1 stale, 1 wrong story, 2 failed; 2 quotes kept, 1 dropped; 1 swap",
                 "Coverage per pick: live 4 outlets / 15 HN pts, shadow 2 outlets / 7 HN pts",
                 "Same fact-check on 2 shared stories: live 1 unsupported, shadow 2; numbers: live 1, shadow 1",
                 "first-round flags live 3 / shadow None; template fallbacks live 1 / shadow 0",
                 "QA: live passed 120 s, shadow failed (RuntimeError: render failed)",
                 "Cost: live $0.25, shadow $1.00; Tavily 0 + 9 credits (month 12/700)"):
        assert part in text, part
    assert "(critic failed)" not in text
    report = compare(live, replace(shadow_rec, error=""), None, None, "S", "H")
    text = summary(report)
    assert "fact-check on shared stories: skipped" in text and "shadow no video" in text
    assert "(month" not in text and "live ? outlets / ? HN pts" in text
    assert "skipped today (budget)" in summary({"date": "2026-09-26", "skipped": "budget"})
    crashed = summary({"date": "2026-09-26", "error": "ValueError: boom"})
    assert "crashed (ValueError: boom)" in crashed and "not affected" in crashed


# --- the shadow run inside the pipeline -----------------------------------------------------

JINA = "https://r.jina.ai/"
PAGE = "ARTICLEMARKER the agent books whole trips on its own. " * 12
# Feed text with enough substance to air (description_problem: 12+ words, 10+ beyond the headline).
SUMMARIES = {
    "OpenAI ships GPT agent": "OpenAI launched an agent inside ChatGPT that browses websites and fills in forms. "
                              "Paying users can ask it to book a table or order groceries, and it checks before buying.",
    "Nvidia unveils inference chip": "Nvidia unveiled a chip made for running AI models rather than training them. "
                                     "Cloud companies expect it to lower what each chatbot answer costs them.",
    "EU passes AI audit rules": "The European Parliament voted for independent audits of high risk AI systems. "
                                "Companies selling hiring or credit scoring tools must prove they are tested for bias.",
    "Anthropic raises funding for Claude": "Anthropic raised new money from investors led by a large tech fund. "
                                           "It pays for the computing power to train and serve future Claude models.",
    "Google Gemini tops math olympiad": "A Gemini model from Google solved most problems from this year's "
                                        "International Mathematical Olympiad, scoring at the level of top students.",
    "Meta open sources Llama": "Meta released its latest Llama model with open weights for researchers and companies. "
                               "Developers can download it and run it on their own servers without paying fees.",
    "Mistral releases coding model": "French startup Mistral released a model that writes and fixes software code. "
                                     "It runs on a single graphics card and plugs into popular code editors.",
    "DeepMind robot learns to cook": "Google DeepMind trained a robot arm to prepare simple meals by watching videos "
                                     "of people cooking, then practicing each step in a simulated kitchen.",
    "Apple adds AI to Siri": "Apple is rebuilding Siri around a large language model in its next iPhone update. "
                             "The assistant will understand follow-up questions and act inside third-party apps.",
    "Microsoft launches Copilot tutor": "Microsoft launched a Copilot mode for students that explains homework step "
                                        "by step instead of giving answers. Schools can switch it on for free this fall.",
}
TITLES = list(SUMMARIES)


class DeskLLM:
    """Picks the given stories as the editor (with their feed summaries), finds no duplicates, fails as the writer
    (so the template writes), finds nothing as critic."""

    name, model = "desk", "desk-1"

    def __init__(self, picks):
        self.picks, self.calls = picks, []

    def json(self, system, user, *, stage, schema=None):
        self.calls.append(stage)
        if stage == "editor":
            return {"stories": [_pick(s.title, s.url, s.summary) for s in self.picks], "alternates": []}
        if stage == "dedupe":
            return {"duplicates": []}
        if stage == "critic":
            return {"segments": []}
        raise RuntimeError(f"{stage}: nothing scripted")


class FakeVoice(SilentVoice):
    name = "fake"  # not "silent", so a live run passes QA


class Reply:
    def __init__(self, status=404, text=""):
        self.status_code, self.text, self.content, self.url = status, text, text.encode(), ""

    def json(self):
        raise ValueError("not json")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def close(self):
        pass


class FakeYouTube:
    name = "youtube"

    def __init__(self, uploads):
        self.uploads = uploads

    def upload(self, video, episode):
        self.uploads.append(video)
        return UploadResult("youtube", "https://youtube.com/shorts/abc")


def _stub_run(monkeypatch, tmp_path, candidates=None, llm=True):
    """A live run with fake news, voice, video, uploader, emails and web (Jina has one article; the rest is 404)."""
    monkeypatch.setenv("SHORTS_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("SHORTS_STATE_DIR", str(tmp_path / "state"))
    news = [_story(t, summary=SUMMARIES[t]) for t in TITLES]
    candidates = news if candidates is None else candidates
    rig = SimpleNamespace(news=news, fetches=[], notes=[], http=[], llms=[], uploads=[], uploaders=[], renders=[],
                          music=[], http_before_shadow=None)

    def fetch(sources):
        rig.fetches.append(sources)
        return candidates

    def fake_http(url, *args, **kwargs):
        rig.http.append(url)
        if url == JINA + news[2].url:
            return Reply(200, f"Title: {news[2].title}\nURL Source: {news[2].url}\nMarkdown Content:\n{PAGE}")
        return Reply()

    def no_session(self, method, url, *args, **kwargs):
        rig.http.append(url)
        raise requests.ConnectionError("no network in tests")

    def build_llm(cfg, usage):
        if cfg.web and rig.http_before_shadow is None:
            rig.http_before_shadow = len(rig.http)
        fake = DeskLLM(news[2:] if cfg.web else news[:8]) if llm else None
        rig.llms.append((cfg.web, fake))
        return fake

    def build_uploader(cfg):
        rig.uploaders.append(cfg.uploader)
        return FakeYouTube(rig.uploads) if cfg.uploader == "youtube" else real_uploader(cfg)

    def render(vo, cards, desk, host, out, preset, music=None, music_volume=0.0):
        rig.renders.append((out, preset))
        rig.music.append(music)
        return out

    real_uploader = pipeline.build_uploader
    monkeypatch.setattr(pipeline, "fetch_all", fetch)
    monkeypatch.setattr(pipeline, "build_llm", build_llm)
    monkeypatch.setattr(pipeline, "build_voice", lambda cfg: FakeVoice())
    monkeypatch.setattr(pipeline, "build_uploader", build_uploader)
    monkeypatch.setattr(pipeline, "notify", lambda notifier, subject, text: rig.notes.append((subject, text)))
    monkeypatch.setattr(pipeline.composer, "render", render)
    monkeypatch.setattr(qa, "media_duration", lambda p: 120.0)
    monkeypatch.setattr(qa, "has_audio", lambda p: True)
    for name in ("get", "post", "head"):  # shorts.web/coverage/sources/research all share this module
        monkeypatch.setattr(requests, name, fake_http)
    monkeypatch.setattr(requests.Session, "request", no_session)
    # allow_no_ai: DeskLLM's writer fails on purpose, and a live run only takes the template script with this on
    rig.cfg = replace(Config.from_env().offline(), sources=["rss"], shadow=True, uploader="youtube", allow_no_ai=True)
    return rig


def _headlines(path):
    return [s["headline"] for s in json.loads(path.read_text())]


def test_shadow_run_writes_its_folder_and_leaves_production_alone(monkeypatch, tmp_path):
    rig = _stub_run(monkeypatch, tmp_path)
    video = pipeline.run(rig.cfg, upload=True)
    run_dir, out = video.parent, video.parent / "shadow"
    assert video == run_dir / "short.mp4" and rig.uploads == [video]  # only production went to YouTube
    assert rig.uploaders == ["youtube"] and not (run_dir / "upload.json").exists()  # the shadow's is local
    assert json.loads((out / "upload.json").read_text())["file"] == "short.mp4"
    for name in ("02-picks.json", "02-research.json", "02-coverage.json", "03-episode.json", "compare.json",
                 "cost.json", "qa.json"):
        assert (out / name).exists(), name
    assert not (run_dir / "02-research.json").exists()  # production ran with the web off
    assert len(rig.fetches) == 1  # the shadow reused production's candidates
    assert [web for web, _ in rig.llms] == [False, True, True]  # production, shadow, cross-check critic
    assert rig.http_before_shadow == 0  # production made no web request
    assert [preset for _, preset in rig.renders] == [rig.cfg.x264_preset, "ultrafast"]
    assert json.loads((out / "02-picks.json").read_text())[0]["body"].startswith("ARTICLEMARKER")  # read via Jina
    assert not any("tavily" in u for u in rig.http)  # no Tavily key, no Tavily call
    assert _headlines(run_dir / "02-picks.json") == TITLES[:8]
    assert _headlines(out / "02-picks.json") == TITLES[2:]
    seen = json.loads((tmp_path / "state" / "seen_urls.json").read_text())
    assert sorted(e["url"] for e in seen) == sorted(s.url for s in rig.news[:8])  # production's picks only
    assert len(json.loads((tmp_path / "state" / "intros.json").read_text())) == 1  # production's intro only
    last = json.loads((tmp_path / "state" / "last_episode.json").read_text())
    assert [s["headline"] for s in last["segments"] if s["kind"] == "story"] == TITLES[:8]
    assert last["stories"] == []  # the state branch is public: the script only, no article text
    report = json.loads((out / "compare.json").read_text())
    assert report["picks"] == {"shared": 6, "live_only": TITLES[:2], "shadow_only": TITLES[8:]}
    assert report["writer"]["live"]["qa_passed"] and report["writer"]["shadow"]["qa_passed"]
    assert report["cross_check"] is None and rig.llms[2][1].calls == []  # nothing verified, so no critic call
    rows = [json.loads(line) for line in (tmp_path / "state" / "shadow.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["picks"]["shared"] == 6
    assert "ARTICLEMARKER" not in json.dumps(rows[0]) and "ARTICLEMARKER" not in json.dumps(report)
    assert len(rig.notes) == 2
    assert rig.notes[0][0].startswith("New episode ready")
    assert "shadow day 1" in rig.notes[1][0] and "Picks: 6 shared" in rig.notes[1][1]


def test_a_crash_inside_the_shadow_run_is_recorded_and_production_is_intact(monkeypatch, tmp_path):
    rig = _stub_run(monkeypatch, tmp_path)

    def render(vo, cards, desk, host, out, preset, **kw):
        if out.parent.name == "shadow":
            raise RuntimeError("encoder crashed")
        return out

    monkeypatch.setattr(pipeline.composer, "render", render)
    video = pipeline.run(rig.cfg, upload=True)
    assert video == video.parent / "short.mp4" and rig.uploads == [video]
    report = json.loads((video.parent / "shadow" / "compare.json").read_text())
    assert report["writer"]["shadow"]["error"] == "RuntimeError: encoder crashed"
    assert report["writer"]["shadow"]["qa_passed"] is None and report["writer"]["live"]["qa_passed"]
    assert len(rig.notes) == 2 and "failed (RuntimeError: encoder crashed)" in rig.notes[1][1]
    seen = json.loads((tmp_path / "state" / "seen_urls.json").read_text())
    assert len(seen) == 8


def test_a_crash_in_the_comparison_is_reported_and_production_is_intact(monkeypatch, tmp_path):
    rig = _stub_run(monkeypatch, tmp_path)

    def broken(*args, **kwargs):
        raise ValueError("comparison broke")

    monkeypatch.setattr(shadow, "compare", broken)
    video = pipeline.run(rig.cfg, upload=True)
    assert video == video.parent / "short.mp4" and rig.uploads == [video]
    report = json.loads((video.parent / "shadow" / "compare.json").read_text())
    assert report["error"] == "ValueError: comparison broke" and report["date"] == date.today().isoformat()
    assert len(rig.notes) == 2 and "crashed (ValueError: comparison broke)" in rig.notes[1][1]


def test_budget_guard_skips_the_shadow_with_no_llm_call(monkeypatch, tmp_path):
    rig = _stub_run(monkeypatch, tmp_path)
    month = date.today().strftime("%Y-%m")
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "spend.json").write_text(json.dumps({month: 17.9}))  # of the $18 monthly cap
    video = pipeline.run(rig.cfg, upload=True)
    out = video.parent / "shadow"
    assert [web for web, _ in rig.llms] == [False]  # no model was even built for the shadow
    assert rig.http_before_shadow is None and rig.http == []
    assert json.loads((out / "compare.json").read_text()) == {"date": date.today().isoformat(), "skipped": "budget"}
    assert not (out / "02-picks.json").exists()
    rows = (tmp_path / "state" / "shadow.jsonl").read_text().splitlines()
    assert [json.loads(r)["skipped"] for r in rows] == ["budget"]
    assert len(rig.notes) == 2 and "skipped today (budget)" in rig.notes[1][1]
    assert json.loads((tmp_path / "state" / "spend.json").read_text()) == {month: 17.9}


def test_production_failing_after_the_fetch_still_gets_its_shadow(monkeypatch, tmp_path):
    rig = _stub_run(monkeypatch, tmp_path)
    monkeypatch.setattr(qa, "media_duration", lambda p: 120.0 if p.parent.name == "shadow" else 10.0)
    with pytest.raises(RuntimeError, match="Quality check failed"):
        pipeline.run(rig.cfg, upload=True)
    [out] = list((tmp_path / "out").glob("*/shadow"))
    report = json.loads((out / "compare.json").read_text())
    assert report["writer"]["live"]["qa_passed"] is False and "Quality check failed" in report["writer"]["live"]["error"]
    assert report["writer"]["shadow"]["qa_passed"] is True
    assert rig.uploads == [] and not (tmp_path / "state" / "seen_urls.json").exists()
    assert not (tmp_path / "state" / "intros.json").exists() and not (tmp_path / "state" / "last_episode.json").exists()
    assert [subject for subject, _ in rig.notes] == ["Duck Desk shadow day 1: web research vs live"]


def _no_shadow(rig, tmp_path):
    assert not (tmp_path / "state" / "shadow.jsonl").exists()
    assert not any("shadow" in subject for subject, _ in rig.notes)
    assert not list((tmp_path / "out").glob("*/shadow"))
    assert all(not web for web, _ in rig.llms)


def test_no_shadow_offline_with_the_web_on_or_when_production_never_fetched(monkeypatch, tmp_path):
    rig = _stub_run(monkeypatch, tmp_path / "offline")
    pipeline.run(replace(Config.from_env().offline(), shadow=True))  # sample news
    _no_shadow(rig, tmp_path / "offline")
    assert len(rig.notes) == 1

    rig = _stub_run(monkeypatch, tmp_path / "web", llm=False)
    video = pipeline.run(replace(rig.cfg, web=True, uploader="local"))
    assert (video.parent / "02-research.json").exists()  # production itself ran the web path
    assert not (tmp_path / "web" / "state" / "shadow.jsonl").exists() and not (video.parent / "shadow").exists()
    assert len(rig.notes) == 1

    rig = _stub_run(monkeypatch, tmp_path / "empty", candidates=[])
    with pytest.raises(RuntimeError, match="No stories"):
        pipeline.run(rig.cfg)
    _no_shadow(rig, tmp_path / "empty")
    assert rig.notes == []


def test_an_interrupted_run_starts_no_shadow(monkeypatch, tmp_path):
    rig = _stub_run(monkeypatch, tmp_path)

    def render(*args, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline.composer, "render", render)
    with pytest.raises(KeyboardInterrupt):
        pipeline.run(rig.cfg, upload=True)
    _no_shadow(rig, tmp_path)
    assert list((tmp_path / "out").glob("*/cost.json"))  # the cost is still saved


def test_config_reads_the_web_and_shadow_switches(monkeypatch):
    for name in ("SHORTS_WEB", "SHORTS_SHADOW", "SHORTS_TAVILY_MONTHLY_CREDITS", "SHORTS_TAVILY_RUN_CREDITS"):
        monkeypatch.delenv(name, raising=False)
    cfg = Config.from_env()
    assert (cfg.web, cfg.shadow, cfg.tavily_monthly_credits, cfg.tavily_run_credits) == (False, False, 700, 25)
    monkeypatch.setenv("SHORTS_WEB", "on")
    monkeypatch.setenv("SHORTS_SHADOW", "on")
    monkeypatch.setenv("SHORTS_TAVILY_MONTHLY_CREDITS", "500")
    monkeypatch.setenv("SHORTS_TAVILY_RUN_CREDITS", "12")
    cfg = Config.from_env()
    assert (cfg.web, cfg.shadow, cfg.tavily_monthly_credits, cfg.tavily_run_credits) == (True, True, 500, 12)
    off = cfg.offline()
    assert (off.web, off.shadow, off.sources) == (False, False, ["sample"])
    monkeypatch.setenv("SHORTS_SHADOW", "off")
    assert not Config.from_env().shadow


def test_offline_run_with_web_and_shadow_switched_on_makes_no_http(tmp_path, monkeypatch):
    calls = []

    def no_network(*args, **kwargs):
        calls.append(args[:2])
        raise AssertionError(f"HTTP in an offline run: {args[:2]}")

    for name in ("get", "post", "head"):
        monkeypatch.setattr(requests, name, no_network)
    monkeypatch.setattr(requests.Session, "request", no_network)
    monkeypatch.setenv("SHORTS_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("SHORTS_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("SHORTS_X264_PRESET", "ultrafast")
    monkeypatch.setenv("SHORTS_WEB", "on")
    monkeypatch.setenv("SHORTS_SHADOW", "on")
    video = pipeline.run(Config.from_env().offline())
    assert calls == []
    assert video.exists() and 45 < media_duration(video) <= 180
    folder = video.parent
    assert not (folder / "shadow").exists() and not (folder / "02-research.json").exists()
    assert not (tmp_path / "state" / "shadow.jsonl").exists()
    for name in ("02-picks.json", "03-episode.json", "qa.json", "upload.json", "cost.json"):
        assert (folder / name).exists(), name
