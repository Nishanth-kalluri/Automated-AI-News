import json
import logging
import sys
import threading
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import requests

from shorts import coverage, sources, web
from shorts.checks import _material, episode_material, lint_episode
from shorts.coverage import Coverage
from shorts.llm import BudgetExceeded, OpenAILLM, SpendLedger, Tool, Usage, capped, strict_object
from shorts.models import Evidence, Segment
from shorts.selection import (AGENT_TOOLS_NOTE, ALTERNATES_NOTE, COVERAGE_NOTE, COVERAGE_TOOL_NOTE, EDITOR_SCHEMA,
                              EDITOR_SYSTEM, EDITOR_WEB_SCHEMA, PICK_SCHEMA, SEARCH_TOOL_NOTE, AgentEditor,
                              HeuristicEditor, LLMEditor, SeenStore, _published, build_editor, pick_stories,
                              pick_with_fallback)
from shorts.web import Tavily, TavilyCredits
from shorts.writer import CRITIC_EVIDENCE_NOTE, CRITIC_SYSTEM, REVIEW_SCHEMA, CriticWriter, TemplateWriter, _story_block
from tests.test_agents import (GOOD_A, GOOD_B, NO_ISSUES, FakeResponses, ScriptedLLM, _openai, _pick, _resp, _script,
                               _stories, _story)

PHASE1_TOOLS_NOTE = """

You have two tools. Use them before you answer; a few calls are enough.
- search_candidates(query): searches today's newsletters and feeds. Use it to find every outlet that covered an
  event (more outlets means a bigger story) and the best link for it. A link must come from today's material.
- aired_lookup(headline): checks whether a story already aired in the last two weeks."""
LIMIT = "error: tool limit reached"
NAMES = ["Anthropic", "Nvidia", "Google", "Meta", "Mistral", "Apple", "Amazon", "Microsoft", "Cohere", "Samsung",
         "Intel", "Baidu"]
PRODUCTS = ["Claude", "Blackwell", "Gemini", "Llama", "Codestral", "Siri", "Nova", "Copilot", "Command", "Gauss",
            "Gaudi", "Ernie"]


class RecordingLLM(ScriptedLLM):
    """ScriptedLLM that also keeps each call's system prompt and schema."""

    def __init__(self, *replies):
        super().__init__(*replies)
        self.systems, self.schemas = [], []

    def json(self, system, user, *, stage, schema=None):
        self.systems.append(system)
        self.schemas.append(schema)
        return super().json(system, user, stage=stage, schema=schema)


class FakeCoverage:
    def __init__(self, rows=None):
        self.rows, self.batches, self.lookups = rows or {}, [], []

    def lookup_many(self, headlines, timeout=30):
        self.batches.append((list(headlines), timeout))
        return {h: self.rows[h] for h in headlines if h in self.rows}

    def lookup(self, headline, url=""):
        self.lookups.append(headline)
        return {"headline": headline, "hn_points": 7, "hn_threads": 1, "news_outlets": 2, "outlets": ["A", "B"]}


def _no_http(*args, **kwargs):
    raise AssertionError("unexpected HTTP call")


def _tavily(monkeypatch, results_for, posts=None):
    """A real Tavily client whose HTTP is a fake: ``results_for(query)`` gives the search results."""
    posts = [] if posts is None else posts

    def post(url, json=None, headers=None, timeout=None):
        posts.append((url, json))
        return SimpleNamespace(status_code=200, json=lambda: {"results": results_for(json["query"])})

    monkeypatch.setattr(web, "requests", SimpleNamespace(post=post, get=_no_http, head=_no_http,
                                                         RequestException=requests.RequestException))
    return Tavily("tvly", TavilyCredits(None))


def _hit(url, title, day="", content="A snippet about the launch."):
    return {"url": url, "title": title, "content": content, "raw_content": content, "published_date": day}


def _answer(stories, alternates=None):
    data = {"stories": stories}
    if alternates is not None:
        data["alternates"] = alternates
    return json.dumps(data)


def _tool_names(call):
    return [t["name"] for t in call["tools"]]


def _outputs(call):
    return {o["call_id"]: o["output"] for o in call["input"]}


def _feed(k):
    return [_story(f"{n} ships {p} update", hours_ago=i + 1) for i, (n, p) in enumerate(zip(NAMES[:k], PRODUCTS[:k]))]


# --- editor with web off: exactly phase 1 ----------------------------------------------------

def test_editor_without_web_sends_the_phase_1_prompt_tools_and_schema(tmp_path):
    letter = _story("TLDR AI issue", kind="newsletter", url="", source="TLDR AI", body="Lab news (https://lab.ai/x)")
    a, b = _story("OpenAI ships GPT agent", hours_ago=2), _story("Nvidia unveils AI chip", hours_ago=5)
    candidates = [letter, b, a]
    answer = _answer([_pick("OpenAI ships GPT agent", a.url)], alternates=[_pick("Nvidia unveils AI chip", b.url)])
    llm, fake = _openai(_resp(text=answer))
    editor = AgentEditor(llm, 30)
    assert not editor.web
    picks = editor.pick(candidates, 1, SeenStore(tmp_path / "seen.json"))
    expected = "\n".join([
        f"Today is {date.today().isoformat()}. Pick exactly 1 stories.\n",
        f"=== NEWSLETTER 1: TLDR AI | TLDR AI issue | {letter.published:%Y-%m-%d %H:%M} UTC ===\n{letter.body}\n",
        "=== FEED HEADLINES (source | published | title | url | snippet) ===",
        f"- Feed | {a.published:%m-%d %H:%M} | {a.title} | {a.url} | {a.summary}",
        f"- Feed | {b.published:%m-%d %H:%M} | {b.title} | {b.url} | {b.summary}",
    ])
    call = fake.calls[0]
    assert call["input"] == expected and "coverage" not in call["input"]
    assert call["instructions"] == EDITOR_SYSTEM + PHASE1_TOOLS_NOTE and AGENT_TOOLS_NOTE == PHASE1_TOOLS_NOTE
    assert _tool_names(call) == ["search_candidates", "aired_lookup"]
    assert call["text"]["format"]["schema"] == EDITOR_SCHEMA == {
        "type": "object", "properties": {"stories": {"type": "array", "items": PICK_SCHEMA}},
        "required": ["stories"], "additionalProperties": False}
    assert [p.headline for p in picks] == ["OpenAI ships GPT agent"] and editor.alternates == []


def test_editors_without_web_and_without_tools_use_the_plain_system_and_schema(tmp_path):
    candidates = [_story("OpenAI ships GPT agent")]
    reply = {"stories": [_pick("OpenAI ships GPT agent", candidates[0].url)]}
    for editor_cls in (AgentEditor, LLMEditor):
        llm = RecordingLLM(reply)
        editor_cls(llm, 30).pick(candidates, 1, SeenStore(tmp_path / "seen.json"))
        assert llm.systems == [EDITOR_SYSTEM] and llm.schemas == [EDITOR_SCHEMA]
        assert "coverage" not in llm.calls[0][1]


# --- editor with web on --------------------------------------------------------------------

def test_coverage_column_is_on_the_top_10_feed_lines_one_per_event(tmp_path):
    feed = _feed(12)
    dup = _story("OpenAI releases GPT-6 Sol model", hours_ago=0.5)
    top = _story("OpenAI launches GPT-6 Sol", hours_ago=0.2)
    candidates = [*feed, dup, top]
    rows = {top.title: {"hn_points": 240, "hn_threads": 3, "news_outlets": 9, "outlets": []},
            feed[0].title: {"hn_points": None, "hn_threads": None, "news_outlets": 3, "outlets": []},
            feed[1].title: {"hn_points": None, "hn_threads": None, "news_outlets": None, "outlets": []}}
    cov = FakeCoverage(rows)
    prompt = LLMEditor(None, 30, cov)._prompt(candidates, 3, SeenStore(tmp_path / "seen.json"))
    titles, timeout = cov.batches[0]
    assert len(cov.batches) == 1 and timeout == 30
    assert titles == [top.title] + [s.title for s in feed[:9]]  # dup skipped, one lookup per event
    lines = {line.split(" | ")[2]: line for line in prompt.splitlines() if line.startswith("- ")}
    assert "=== FEED HEADLINES (source | published | title | url | snippet | coverage) ===" in prompt
    assert lines[top.title].endswith(f"{top.url} | {top.summary} | coverage: HN 240 pts, 9 outlets on Google News")
    assert lines[feed[0].title].endswith(" | coverage: HN unknown, 3 outlets on Google News")
    assert lines[feed[1].title].endswith(" | coverage unknown") and lines[feed[5].title].endswith(" | coverage unknown")
    assert "coverage" not in lines[dup.title] and "coverage" not in lines[feed[9].title]
    assert sum("coverage" in line for line in lines.values()) == 10


def test_coverage_column_says_unknown_when_both_backends_are_down(tmp_path, monkeypatch):
    def down(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(sources, "requests", SimpleNamespace(get=down))
    monkeypatch.setattr(coverage, "requests", SimpleNamespace(get=down, RequestException=requests.RequestException))
    feed = _feed(12)
    prompt = LLMEditor(None, 30, Coverage(30))._prompt(feed, 3, SeenStore(tmp_path / "seen.json"))
    lines = [line for line in prompt.splitlines() if line.startswith("- ")]
    assert all(line.endswith(" | coverage unknown") for line in lines[:10])
    assert not any("coverage" in line for line in lines[10:]) and len(lines) == 12


def test_llm_editor_with_coverage_adds_the_note_but_keeps_the_schema(tmp_path):
    candidates = [_story("OpenAI ships GPT agent")]
    llm = RecordingLLM({"stories": [_pick("OpenAI ships GPT agent", candidates[0].url)]})
    LLMEditor(llm, 30, FakeCoverage()).pick(candidates, 1, SeenStore(tmp_path / "seen.json"))
    assert llm.systems == [EDITOR_SYSTEM + COVERAGE_NOTE] and llm.schemas == [EDITOR_SCHEMA]
    assert llm.calls[0][1].count(" | coverage unknown") == 1


def test_web_editor_offers_coverage_and_web_search_and_keeps_three_alternates(tmp_path, monkeypatch, caplog):
    feed = _feed(6)
    tavily = _tavily(monkeypatch, lambda q: [])
    alternates = [_pick(s.title, s.url) for s in feed[1:5]]
    llm, fake = _openai(_resp(text=_answer([_pick(feed[0].title, feed[0].url)], alternates)))
    editor = AgentEditor(llm, 30, coverage=FakeCoverage(), tavily=tavily)
    picks = editor.pick(feed, 1, SeenStore(tmp_path / "seen.json"))
    call = fake.calls[0]
    assert _tool_names(call) == ["search_candidates", "aired_lookup", "coverage", "web_search"]
    assert call["text"]["format"]["schema"] == EDITOR_WEB_SCHEMA
    assert set(EDITOR_WEB_SCHEMA["required"]) == {"stories", "alternates"}
    system = call["instructions"]
    assert system == (EDITOR_SYSTEM + COVERAGE_NOTE + AGENT_TOOLS_NOTE.replace("two tools", "these tools")
                      + COVERAGE_TOOL_NOTE + SEARCH_TOOL_NOTE + ALTERNATES_NOTE)
    assert " | coverage" in call["input"]
    assert [p.headline for p in picks] == [feed[0].title]
    assert [a.headline for a in editor.alternates] == [s.title for s in feed[1:4]]
    assert editor.alternates[0].url == feed[1].url and editor.alternates[0].published == feed[1].published
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="shorts.selection"):
        llm, fake = _openai(_resp(text=_answer([_pick(feed[0].title, feed[0].url)], alternates[:1])))
        editor = AgentEditor(llm, 30, coverage=FakeCoverage(), tavily=tavily)
        editor.pick(feed, 1, SeenStore(tmp_path / "seen.json"))
    assert [a.headline for a in editor.alternates] == [feed[1].title]
    assert not [r for r in caplog.records if "editor returned" in r.getMessage()]  # alternates are quiet


def test_web_editor_without_a_usable_tavily_has_no_web_search_tool(tmp_path, monkeypatch):
    feed = _feed(2)
    tavily = _tavily(monkeypatch, lambda q: [])
    tavily.disabled = "HTTP 401"
    llm, fake = _openai(_resp(text=_answer([_pick(feed[0].title, feed[0].url)], [])))
    editor = AgentEditor(llm, 30, tavily=tavily)
    editor.pick(feed, 1, SeenStore(tmp_path / "seen.json"))
    call = fake.calls[0]
    assert _tool_names(call) == ["search_candidates", "aired_lookup"]
    assert call["instructions"] == EDITOR_SYSTEM + AGENT_TOOLS_NOTE + ALTERNATES_NOTE
    assert call["text"]["format"]["schema"] == EDITOR_WEB_SCHEMA and "coverage" not in call["input"]


def test_web_editor_without_tools_asks_for_alternates_and_repairs_with_the_old_schema(tmp_path):
    feed = _feed(4)
    draft = {"stories": [_pick(feed[0].title, feed[0].url), _pick(feed[0].title, feed[0].url)],
             "alternates": [_pick(feed[3].title, feed[3].url)]}
    fixed = {"stories": [_pick(feed[0].title, feed[0].url), _pick(feed[1].title, feed[1].url)]}
    llm = RecordingLLM(draft, fixed)
    editor = AgentEditor(llm, 30, coverage=FakeCoverage())
    picks = editor.pick(feed, 2, SeenStore(tmp_path / "seen.json"))
    assert [c[0] for c in llm.calls] == ["editor", "editor-repair-1"]
    assert llm.systems[0] == EDITOR_SYSTEM + COVERAGE_NOTE + ALTERNATES_NOTE
    assert llm.schemas == [EDITOR_WEB_SCHEMA, EDITOR_SCHEMA]
    assert [p.headline for p in picks] == [feed[0].title, feed[1].title]
    assert [a.headline for a in editor.alternates] == [feed[3].title]  # the draft's alternates stay


def test_a_link_from_web_search_passes_settle_and_takes_the_hit_date(tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)
    primary = "https://www.lab.ai/blog/agent-launch"
    fresh = "https://ai-daily.example/openai-agent-rival"
    tavily = _tavily(monkeypatch, lambda q: [
        _hit(primary, "Lab launches travel agent AI", f"{yesterday}T08:00:00Z"),
        _hit(fresh, "OpenAI rival ships AI agent model", today.isoformat()),
        _hit("https://nodate.example/post", "Undated AI post")])
    feed = _feed(3)
    candidates = list(feed)
    answer = _answer([_pick("Lab launches travel agent", primary + "?utm_source=tavily"),
                      _pick("Invented AI launch", "https://invented.example/story")], [])
    llm, fake = _openai(_resp(calls=[("web_search", {"query": "lab travel agent"})], rid="r1"),
                        _resp(text=answer, rid="r2"))
    seen = SeenStore(tmp_path / "seen.json")
    before = datetime.now(timezone.utc)
    picks = pick_with_fallback(AgentEditor(llm, 30, max_repairs=0, tavily=tavily), candidates, 3, seen, 30)
    after = datetime.now(timezone.utc)
    rows = _outputs(fake.calls[1])["call0"]
    assert rows.startswith(f"1. Lab launches travel agent AI | {primary} | {yesterday}")
    assert "date unknown" in rows and "A snippet about the launch." in rows
    webs = [c for c in candidates if c.kind == "web"]
    assert [c.url for c in webs] == [primary, fresh, "https://nodate.example/post"]
    assert webs[0].source == "lab.ai" and webs[0].summary == "A snippet about the launch."
    assert webs[0].published == datetime(yesterday.year, yesterday.month, yesterday.day, 23, 59, tzinfo=timezone.utc)
    assert before <= webs[1].published <= after and before <= webs[2].published <= after  # capped at now
    assert picks[0].headline == "Lab launches travel agent" and picks[0].url.startswith(primary)
    assert picks[0].published == webs[0].published and picks[0].source == "lab.ai" and "lab.ai" in picks[0].outlets
    assert picks[1].headline == "Invented AI launch" and picks[1].url == ""  # made-up link still dropped
    assert picks[2] in feed  # topped up from the feed, never from the web hits
    assert not any(p in webs for p in picks)
    assert not [s for s in pick_stories(candidates, 10, 30) if s.kind == "web"]
    assert not [s for s in HeuristicEditor(30).pick(candidates, len(candidates), seen) if s.kind == "web"]
    prompt = LLMEditor(None, 30)._prompt(candidates, 3, seen)
    assert fresh not in prompt and primary not in prompt and "OpenAI rival ships" not in prompt


def test_search_hits_already_in_the_material_are_not_added_twice(tmp_path, monkeypatch):
    feed = _feed(2)
    tavily = _tavily(monkeypatch, lambda q: [_hit(feed[0].url, "Same story"), _hit("https://lab.ai/new", "New")])
    editor = AgentEditor(ScriptedLLM(), 30, tavily=tavily)
    candidates = list(feed)
    editor._search(candidates, "first")
    editor._search(candidates, "again")
    assert [c.url for c in candidates] == [feed[0].url, feed[1].url, "https://lab.ai/new"]
    assert candidates[0].kind == "article"


def test_search_hit_date_is_the_end_of_that_day_capped_at_now():
    now = datetime(2026, 9, 26, 15, 30, tzinfo=timezone.utc)
    assert _published("2026-09-20", now) == datetime(2026, 9, 20, 23, 59, tzinfo=timezone.utc)
    assert _published("2026-09-26", now) == now
    assert _published("2026-10-02", now) == now
    assert _published("", now) == now and _published("last week", now) == now


def test_to_stories_matches_web_hits_like_feed_articles():
    hit = _story("Lab post", url="https://lab.ai/post", kind="web", source="lab.ai", hours_ago=20)
    letter = _story("Letter", kind="newsletter", url="https://letter.example/issue", source="TLDR AI")
    data = {"stories": [_pick("Lab ships agent", "https://www.lab.ai/post/"),
                        _pick("From the letter", "https://letter.example/issue")]}
    picks = LLMEditor.to_stories(data, [hit, letter], 2)
    assert picks[0].published == hit.published and picks[0].source == "lab.ai"
    assert picks[0].outlets == ["TLDR AI", "lab.ai"]
    assert picks[1].source == "TLDR AI" and picks[1].published > hit.published  # a newsletter never matches


def test_editor_tool_limits_are_10_coverage_and_3_web_search_calls(tmp_path, monkeypatch):
    posts = []
    tavily = _tavily(monkeypatch, lambda q: [_hit(f"https://lab.ai/{q}", q)], posts)
    cov = FakeCoverage()
    calls = ([("web_search", {"query": f"q{i}"}) for i in range(3)]
             + [("coverage", {"headline": f"Event {i}"}) for i in range(11)]
             + [("web_search", {"query": "q3"})])
    feed = _feed(2)
    llm, fake = _openai(_resp(calls=calls, rid="r1"), _resp(text=_answer([_pick(feed[0].title, feed[0].url)], [])))
    AgentEditor(llm, 30, coverage=cov, tavily=tavily).pick(feed, 1, SeenStore(tmp_path / "seen.json"))
    out = _outputs(fake.calls[1])
    assert [not out[f"call{i}"].startswith(LIMIT) for i in range(15)] == [True] * 13 + [False, False]
    assert json.loads(out["call3"])["hn_points"] == 7 and out["call2"].startswith("1. q2 | https://lab.ai/q2")
    assert len(posts) == 3 and len(cov.lookups) == 10


def test_build_editor_passes_the_web_tools_through():
    cov, tavily = FakeCoverage(), SimpleNamespace(usable=True)
    assert isinstance(build_editor(None, 30, agents=True, coverage=cov), HeuristicEditor)
    agent = build_editor(ScriptedLLM(), 30, agents=True, max_repairs=1, coverage=cov, tavily=tavily)
    assert isinstance(agent, AgentEditor) and agent.web and agent.max_repairs == 1
    assert agent.coverage is cov and agent.tavily is tavily and agent.base.coverage is cov
    plain = build_editor(ScriptedLLM(), 30, coverage=cov)
    assert type(plain) is LLMEditor and plain.coverage is cov
    assert not build_editor(ScriptedLLM(), 30, agents=True).web


# --- writer and critic with evidence -------------------------------------------------------

def _researched():
    a, b = _stories()
    a.body = "ARTICLE BODY that the writer must not see."
    a.first_reported = "2026-09-24"
    a.evidence = [Evidence("The agent books travel in 3 steps.", "https://www.lab.ai/post"),
                  Evidence("It asks before paying.", "not a link")]
    b.body = "Chip article text."
    return [a, b]


def _phase1_block(stories):
    out = []
    for i, s in enumerate(stories, 1):
        part = f"STORY {i}: {s.headline or s.title}\nCovered by: {', '.join(s.outlets or [s.source])}\n"
        part += f"Key fact: {s.key_fact}\nSummary: {s.summary}\n"
        if s.body:
            part += f"Article text:\n{s.body}\n"
        out.append(part)
    return "\n".join(out)


def test_story_block_with_evidence_shows_quotes_and_first_report_instead_of_the_article():
    block_a, block_b = _story_block(_researched()).split("\n\nSTORY 2: ")
    lines = block_a.splitlines()
    assert "First reported: 2026-09-24" in lines
    ev = lines.index("Evidence (verbatim quotes from the sources):")
    assert lines.index(f"Summary: {_stories()[0].summary}") < ev
    assert lines[ev + 1:] == ['[E1] "The agent books travel in 3 steps." (lab.ai)', '[E2] "It asks before paying." (source)']
    assert "Article text" not in block_a and "ARTICLE BODY" not in block_a
    assert "Article text:\nChip article text." in block_b and "Evidence" not in block_b
    assert "First reported" not in block_b


def test_story_block_keeps_the_article_when_fewer_than_two_quotes_were_checked():
    a, b = _researched()
    a.evidence = a.evidence[:1]
    block = _story_block([a, b]).split("\n\nSTORY 2: ")[0]
    assert '[E1] "The agent books travel in 3 steps." (lab.ai)' in block
    assert block.endswith("Article text:\nARTICLE BODY that the writer must not see.")


def test_story_block_without_evidence_is_byte_identical_to_phase_1():
    stories = _stories()
    stories[0].body = "Full article text\nwith two lines."
    stories[1].outlets = ["TLDR AI", "The Verge"]
    assert _story_block(stories) == _phase1_block(stories)
    assert _story_block(_stories()) == _phase1_block(_stories())


def test_critic_gets_the_evidence_note_only_when_a_story_has_evidence():
    llm = RecordingLLM(_script(GOOD_A, GOOD_B), NO_ISSUES)
    CriticWriter(llm, llm, "Show", "Host", max_repairs=0).write(_stories())
    assert llm.systems[1] == CRITIC_SYSTEM and llm.schemas[1] == REVIEW_SCHEMA
    stories = _researched()
    llm = RecordingLLM(_script(GOOD_A, GOOD_B), NO_ISSUES)
    CriticWriter(llm, llm, "Show", "Host", max_repairs=0).write(stories)
    (_, writer_prompt), (stage, critic_prompt) = llm.calls
    assert stage == "critic" and llm.systems[1] == CRITIC_SYSTEM + CRITIC_EVIDENCE_NOTE
    assert llm.schemas[1] == REVIEW_SCHEMA and CRITIC_EVIDENCE_NOTE not in llm.systems[0]
    for prompt in (writer_prompt, critic_prompt):  # the writer and the critic see the same material
        assert _story_block(stories) in prompt and "ARTICLE BODY" not in prompt


def test_lint_accepts_a_number_that_only_the_evidence_has():
    story = _story("Lab raises money", summary="The lab raised a new funding round.")
    story.headline = story.title
    story.evidence = [Evidence("The lab raised $40 billion from investors.", "https://lab.ai/post")]
    assert "$40 billion" in _material(story) and "$40 billion" in episode_material([story])
    ep = TemplateWriter("Show", "Host").write([story])
    ep.segments[1] = Segment("story", "The lab raised 40 billion dollars in a new funding round from investors.",
                             "Lab raises money")
    assert not [i for i in lint_episode(ep, [story]) if i.code == "unsupported_number"]
    story.evidence = []
    assert [i.code for i in lint_episode(ep, [story]) if i.code == "unsupported_number"] == ["unsupported_number"]


def test_critic_writer_report_records_each_round():
    hype_a = GOOD_A.replace("The lab shipped", "The lab shipped a revolutionary")
    revision = {"intro": "", "outro": "",
                "segments": [{"story": 1, "headline": "Lab ships agent", "key_fact": "3 steps", "text": GOOD_A}]}
    llm = ScriptedLLM(_script(hype_a, GOOD_B), NO_ISSUES, revision, NO_ISSUES)
    writer = CriticWriter(llm, llm, "Show", "Host")
    writer.write(_stories())
    assert [(r["fatal"], r["critic"]) for r in writer.report["rounds"]] == [(1, 0), (0, 0)]
    assert writer.report["fallbacks"] == [] and writer.report["critic_errors"] == 0
    claim_a = GOOD_A.replace("asks before paying", "pays for everything itself")
    flagged = {"intro": [], "outro": [], "segments": [{"story": 1, "unsupported": ["pays for everything itself"]}]}
    no_change = {"intro": "", "outro": "", "segments": []}
    llm.replies = [_script(claim_a, GOOD_B), flagged, no_change, flagged]
    writer.max_repairs = 1
    writer.write(_stories())
    assert [(r["fatal"], r["critic"]) for r in writer.report["rounds"]] == [(0, 1), (0, 1)]  # a fresh report
    assert writer.report["fallbacks"] == [{"part": "story 1", "why": "not supported by the material: "
                                                                     "pays for everything itself"}]
    llm.replies = [_script(GOOD_A, GOOD_B), RuntimeError("critic down")]
    writer.max_repairs = 0
    writer.write(_stories())
    assert writer.report == {"rounds": [{"fatal": 0, "critic": 0, "problems": []}], "fallbacks": [],
                             "critic_errors": 1}


def test_critic_writer_report_lists_intro_outro_story_and_title_fallbacks():
    script = _script(GOOD_A, GOOD_B)
    script["title"] = "OpenAI raises 400 billion dollars"
    flagged = {"intro": ["teases a third story"], "outro": ["claims the show is on every night"],
               "segments": [{"story": 2, "unsupported": ["cheaper serving"]}]}
    llm = ScriptedLLM(script, flagged)
    writer = CriticWriter(llm, llm, "Show", "Host", max_repairs=0)
    ep = writer.write(_stories())
    assert [(r["fatal"], r["critic"]) for r in writer.report["rounds"]] == [(0, 3)]
    assert len(writer.report["rounds"][0]["problems"]) == 3
    assert [f["part"] for f in writer.report["fallbacks"]] == ["intro", "outro", "story 2", "title"]
    template = TemplateWriter("Show", "Host")
    assert ep.segments[0].text == template.intro(2).text and ep.segments[-1].text == template.outro().text
    assert ep.title == template.write(_stories()).title


def test_each_part_that_falls_back_is_recorded_once():
    script = _script(GOOD_A, GOOD_B.replace("2 times", "5 times"))  # the rules and the critic both catch it
    script["intro"] = "A revolutionary day with 2 AI stories."  # a rule-level intro problem
    flagged = {"intro": [], "outro": [], "segments": [{"story": 2, "unsupported": ["5 times faster"]}]}
    llm = ScriptedLLM(script, flagged)
    writer = CriticWriter(llm, llm, "Show", "Host", max_repairs=0)
    ep = writer.write(_stories())
    assert [(r["fatal"], r["critic"]) for r in writer.report["rounds"]] == [(2, 1)]
    assert ep.story_segments[1].text.startswith("Chip is faster. The new chip is 2 times faster")
    assert [f["part"] for f in writer.report["fallbacks"]] == ["intro", "story 2"]


# --- llm: usage under threads, capped tools, workers and the reserve ------------------------

def _in_threads(fn, args_list):
    """Runs fn(*args) in one thread each, all started together, with frequent thread switches."""
    start = threading.Barrier(len(args_list), timeout=5)

    def run(*args):
        start.wait()
        fn(*args)

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=run, args=args) for args in args_list]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(old)


def test_usage_totals_and_the_ledger_are_exact_under_threads(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.json")
    usage = Usage(on_add=ledger.add)
    _in_threads(lambda: [usage.add("research", "gpt-5", 1000, 100) for _ in range(200)], [()] * 8)
    each = (1000 * 1.25 + 100 * 10) / 1e6
    assert len(usage.calls) == 1600 and usage.total_usd == pytest.approx(1600 * each)
    assert ledger.this_month() == pytest.approx(1600 * each)
    assert SpendLedger(tmp_path / "spend.json").this_month() == pytest.approx(usage.total_usd)


def test_capped_tools_share_one_budget():
    ran = []
    schema = strict_object({"x": {"type": "string"}})
    a, b = capped([Tool("a", "first", schema, lambda x: ran.append(("a", x)) or "A"),
                   Tool("b", "second", schema, lambda x: ran.append(("b", x)) or "B")], 3)
    assert a.spec()["name"] == "a" and a.spec()["parameters"] == schema and b.description == "second"
    results = [a.call('{"x": "1"}'), b.call('{"x": "2"}'), a.call('{"x": "3"}'), b.call('{"x": "4"}'),
               a.call('{"x": "5"}')]
    assert results[:3] == ["A", "B", "A"] and all(r.startswith(LIMIT) for r in results[3:])
    assert ran == [("a", "1"), ("b", "2"), ("a", "3")]
    count = []
    (tool,) = capped([Tool("t", "d", schema, lambda x: count.append(x) or "ok")], 100)
    _in_threads(lambda: [tool.call('{"x": "y"}') for _ in range(50)], [()] * 8)
    assert len(count) == 100


def test_worker_copies_keep_their_own_model_and_fallback_state():
    def model_error():
        return type("NotFoundError", (Exception,), {"status_code": 404})("The model gpt-6-sol does not exist")

    usage = Usage()
    queue = [FakeResponses(model_error(), _resp(text='{"w": 1}')), FakeResponses(_resp(text='{"w": 2}'))]
    options = []

    def with_options(**kwargs):
        options.append(kwargs)
        return SimpleNamespace(responses=queue.pop(0))

    base = OpenAILLM("", "gpt-5", usage, client=SimpleNamespace(responses=FakeResponses(), with_options=with_options))
    sol = base.with_model("gpt-6-sol")
    w1, w2 = sol.worker(), sol.worker(timeout=10, max_retries=0, reserve_usd=0.5)
    assert options == [{"timeout": 45.0, "max_retries": 1}, {"timeout": 10, "max_retries": 0}]
    assert w1.usage is usage and w2.usage is usage and w1.client is not w2.client
    assert (w1.reserve_usd, w2.reserve_usd, sol.reserve_usd) == (0.0, 0.5, 0.0)
    assert w1.json("s", "u", stage="research-1") == {"w": 1}
    assert (w1.model, w1.fallback_model) == ("gpt-5", "")
    assert (w2.model, w2.fallback_model) == ("gpt-6-sol", "gpt-5") == (sol.model, sol.fallback_model)
    assert w2.json("s", "u", stage="research-2") == {"w": 2}
    assert [c["model"] for c in usage.calls] == ["gpt-5", "gpt-6-sol"]  # the rejected call isn't billed
    plain, _ = _openai()
    assert plain.worker().client is plain.client  # a client without with_options is shared


def test_parallel_workers_each_fall_back_once_without_racing():
    def model_error():
        return type("NotFoundError", (Exception,), {"status_code": 404})("The model gpt-6-luna does not exist")

    clients = [SimpleNamespace(responses=FakeResponses(model_error(), _resp(text='{"ok": true}'))) for _ in range(8)]
    base = OpenAILLM("", "gpt-5", Usage(), client=SimpleNamespace(responses=FakeResponses(),
                                                                  with_options=lambda **kw: clients.pop()))
    workers = [base.with_model("gpt-6-luna").worker() for _ in range(8)]
    results, errors = [], []

    def run(w):
        try:
            results.append(w.json("s", "u", stage="research"))
        except Exception as exc:  # the test wants to see any error
            errors.append(exc)

    _in_threads(run, [(w,) for w in workers])
    assert errors == [] and results == [{"ok": True}] * 8
    assert all(w.model == "gpt-5" for w in workers) and len(base.usage.calls) == 8


def test_the_reserve_refuses_calls_that_would_eat_the_writers_share():
    usage = Usage(run_cap_usd=1.0)
    usage.add("editor", "gpt-5", 480_000, 10_000)  # $0.70
    llm, fake = _openai(_resp(text='{"a": 1}'), _resp(text='{"b": 2}'), usage=usage)
    with pytest.raises(BudgetExceeded, match=r"keeping \$0\.35"):
        llm.worker(reserve_usd=0.35).json("s", "u", stage="research")
    assert fake.calls == []
    assert llm.worker(reserve_usd=0.25).json("s", "u", stage="research") == {"a": 1}
    assert llm.json("s", "u", stage="writer") == {"b": 2}  # the writer itself keeps no reserve
    assert usage.over_budget(0.0) is False and usage.over_budget(1.0) is True
    month = Usage(month_cap_usd=10.0, month_spent_usd=9.5)
    month.add("editor", "gpt-5", 160_000, 0)  # $0.20
    assert not month.over_budget() and not month.over_budget(0.2) and month.over_budget(0.35)
    month.check("research", 0.2)
    with pytest.raises(BudgetExceeded):
        month.check("research", 0.35)
