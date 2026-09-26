import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from shorts import qa, sources, voice
from shorts.checks import TARGET_MAX_SECONDS, check_picks, lint_episode, norm_url, predicted_seconds
from shorts.composer import CAPTION_MAX_W, caption_chunks, caption_width, captions_ass
from shorts.llm import BudgetExceeded, OpenAILLM, SpendLedger, Usage
from shorts.models import Episode, Segment, Story, Voiceover, Word
from shorts.selection import AgentEditor, SeenStore, pick_with_fallback
from shorts.upload import youtube_title
from shorts.visuals import CHIP_TEXT_MAX_W, chip_text
from shorts.writer import CriticWriter, TemplateWriter, trim_to_fit

NOW = datetime.now(timezone.utc)


def _story(title, url=None, hours_ago=1, kind="article", summary="An AI model from OpenAI", body="",
           source="Feed"):
    return Story(title=title, url=url if url is not None else f"https://news.example/{title.replace(' ', '-')}",
                 source=source, published=NOW - timedelta(hours=hours_ago), summary=summary, kind=kind, body=body)


def _pick(headline, url, summary="It happened today."):
    return {"headline": headline, "summary": summary, "key_fact": "", "url": url, "outlets": ["TLDR AI"], "why": ""}


class ScriptedLLM:
    """Replies in order; records (stage, user prompt) for every call. No run_tools, like Anthropic."""

    name, model = "scripted", "scripted-1"

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def json(self, system, user, *, stage, schema=None):
        self.calls.append((stage, user))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply(stage, user) if callable(reply) else reply


def _resp(text="", calls=(), rid="r1", inp=1000, out=100, cached=0):
    output = [SimpleNamespace(type="function_call", name=name, arguments=json.dumps(args), call_id=f"call{i}")
              for i, (name, args) in enumerate(calls)]
    usage = SimpleNamespace(input_tokens=inp, output_tokens=out,
                            input_tokens_details=SimpleNamespace(cached_tokens=cached))
    return SimpleNamespace(id=rid, output=output, output_text=text, usage=usage)


class FakeResponses:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _openai(*replies, model="gpt-5", usage=None):
    fake = FakeResponses(*replies)
    return OpenAILLM("", model, usage or Usage(), client=SimpleNamespace(responses=fake)), fake


# --- pick checks and the editor agent -------------------------------------------------------

def test_check_picks_flags_duplicates_repeats_made_up_links_stale_and_samples():
    candidates = [_story("OpenAI ships GPT agent"), _story("Nvidia chip"),
                  _story("Letter", kind="newsletter", url="", body="Read more (https://lab.ai/post?utm_source=x)")]
    picks = [
        _story("OpenAI ships GPT agent"),
        _story("OpenAI ships a GPT agent"),                        # same event again
        _story("Lab post", url="https://lab.ai/post/"),              # link from the newsletter text: fine
        _story("Invented", url="https://made.up/story"),            # link not in today's material
        _story("Old news", hours_ago=100),
        _story("Aired before", url="https://news.example/aired"),
        _story("Fake", url="https://example.com/sample/1"),
    ]
    issues = check_picks(picks, candidates, 8, {"https://news.example/aired"}, [], max_age_hours=30)
    codes = {(i.code, i.index) for i in issues}
    assert ("duplicate", 1) in codes and ("url_not_in_sources", 3) in codes and ("stale", 4) in codes
    assert ("already_aired", 5) in codes and ("sample", 6) in codes and ("too_few", None) in codes
    assert not [i for i in issues if i.index in (0, 2) and i.fatal]
    assert norm_url("https://www.Lab.ai/post/?utm_source=x#top") == "https://lab.ai/post"


def test_agent_editor_sends_only_failing_slots_back_and_keeps_good_picks(tmp_path):
    candidates = [_story("OpenAI ships GPT agent"), _story("Nvidia unveils AI chip"), _story("EU AI rules")]
    first = {"stories": [_pick("OpenAI ships GPT agent", candidates[0].url),
                         _pick("OpenAI ships a GPT agent", candidates[0].url)]}
    fixed = {"stories": [_pick("OpenAI ships GPT agent", candidates[0].url),
                         _pick("Nvidia unveils AI chip", candidates[1].url)]}
    llm = ScriptedLLM(first, fixed)
    picks = AgentEditor(llm, 30).pick(candidates, 2, SeenStore(tmp_path / "seen.json"))
    assert [p.headline for p in picks] == ["OpenAI ships GPT agent", "Nvidia unveils AI chip"]
    stage, prompt = llm.calls[1]
    assert stage == "editor-repair-1" and "PROBLEMS TO FIX" in prompt and "same event as story 1" in prompt
    assert picks[1].source == "Feed" and picks[1].published == candidates[1].published  # matched the feed item


def test_bad_editor_picks_are_dropped_or_fixed_and_topped_up(tmp_path):
    seen = SeenStore(tmp_path / "seen.json")
    aired = _story("Anthropic ships Claude agent")
    aired.headline = aired.title
    seen.add([aired])
    candidates = [_story("OpenAI model launch"), _story("Nvidia AI chip"), _story("Google Gemini update")]
    reply = {"stories": [_pick("OpenAI model launch", "https://invented.link/x"),
                         _pick("Anthropic ships Claude agent", "https://news.example/other"),
                         _pick("OpenAI model launch again", candidates[0].url)]}
    picks = pick_with_fallback(AgentEditor(ScriptedLLM(reply, reply, reply), 30, max_repairs=2),
                               candidates, 3, seen, 30)
    assert len(picks) == 3
    assert picks[0].headline == "OpenAI model launch" and picks[0].url == ""  # made-up link removed, story kept
    assert "Anthropic ships Claude agent" not in [p.headline for p in picks]  # already aired
    assert {"Nvidia AI chip", "Google Gemini update"} <= {p.headline for p in picks}  # topped up


def test_editor_tool_loop_calls_tools_and_returns_the_answer(tmp_path):
    candidates = [_story("OpenAI ships GPT agent"), _story("Letter", kind="newsletter", url="", source="TLDR AI",
                                                           body="OpenAI GPT agent books travel (https://o.ai/a)")]
    answer = json.dumps({"stories": [_pick("OpenAI ships GPT agent", "https://o.ai/a")]})
    llm, fake = _openai(_resp(calls=[("search_candidates", {"query": "OpenAI GPT agent"}),
                                     ("aired_lookup", {"headline": "OpenAI ships GPT agent"})], rid="r1"),
                        _resp(text=answer, rid="r2"))
    picks = AgentEditor(llm, 30).pick(candidates, 1, SeenStore(tmp_path / "seen.json"))
    assert picks[0].url == "https://o.ai/a"
    second = fake.calls[1]
    assert second["previous_response_id"] == "r1" and second["tool_choice"] == "auto"
    outputs = {o["call_id"]: json.loads(o["output"]) for o in second["input"]}
    assert outputs["call0"]["outlets"] == 2 and outputs["call1"] == {"aired": []}
    assert fake.calls[0]["text"]["format"]["type"] == "json_schema"
    assert llm.usage.calls[0]["usd"] == pytest.approx((1000 * 1.25 + 100 * 10) / 1e6)


def test_tool_loop_forces_an_answer_on_the_last_turn():
    looping = _resp(calls=[("search_candidates", {"query": "x"})])
    llm, fake = _openai(looping, looping, _resp(text='{"stories": []}'))
    from shorts.llm import Tool, strict_object
    tool = Tool("search_candidates", "d", strict_object({"query": {"type": "string"}}), lambda query: "nothing")
    assert llm.run_tools("s", "u", [tool], stage="editor", max_turns=2) == {"stories": []}
    assert [c.get("tool_choice") for c in fake.calls] == [None, "auto", "none"]


def test_unavailable_model_falls_back_once():
    err = type("NotFoundError", (Exception,), {"status_code": 404})("The model gpt-6-sol does not exist")
    base, fake = _openai(_resp(text='{"ok": true}'))
    llm = base.with_model("gpt-6-sol")
    fake.replies.insert(0, err)
    assert llm.json("s", "u", stage="writer") == {"ok": True}
    assert llm.model == "gpt-5" and llm.usage.calls[0]["model"] == "gpt-5"


def test_budget_cap_stops_calls_and_editor_falls_back(tmp_path):
    usage = Usage(run_cap_usd=0.01)
    usage.add("editor", "gpt-5", 100_000, 1000)  # $0.135, over the cap
    llm, fake = _openai(_resp(text="{}"), usage=usage)
    with pytest.raises(BudgetExceeded):
        llm.json("s", "u", stage="writer")
    assert fake.calls == []
    picks = pick_with_fallback(AgentEditor(llm, 30), [_story("OpenAI model"), _story("Nvidia AI chip")], 2,
                               SeenStore(tmp_path / "seen.json"), 30)
    assert len(picks) == 2


def test_spend_ledger_carries_the_month_across_runs(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.json")
    ledger.add(12.5)
    usage = Usage(month_cap_usd=18, month_spent_usd=SpendLedger(tmp_path / "spend.json").this_month())
    usage.add("writer", "gpt-5", 5_000_000, 0)  # $6.25
    assert usage.over_budget()


# --- writer agent ----------------------------------------------------------------------------

def _stories():
    a = _story("Lab ships agent", summary="The lab shipped an agent that books travel in 3 steps.")
    b = _story("Chip is faster", summary="The new chip is 2 times faster at inference.")
    for s in (a, b):
        s.headline = s.title
    return [a, b]


def _script(a_text, b_text):
    return {"title": "AI today", "description": "Two stories.", "tags": ["ai"], "intro": "Hello pond, two stories.",
            "segments": [{"headline": "Lab ships agent", "key_fact": "3 steps", "text": a_text},
                         {"headline": "Chip is faster", "key_fact": "2x faster", "text": b_text}],
            "outro": "That's the news from the pond. See you tomorrow!"}


GOOD_A = "The lab shipped an agent that books your travel in 3 steps. You say where and when, it compares options " \
         "and asks before paying. Why it matters: assistants are starting to finish whole errands for you."
GOOD_B = "A new chip runs AI models 2 times faster at inference. Serving chatbots gets cheaper, and cheaper serving " \
         "usually means more free features for users soon. Watch for the big clouds to adopt it first this year."
NO_ISSUES = {"segments": []}


def test_critic_writer_rewrites_only_the_failing_segment():
    hype_a = GOOD_A.replace("The lab shipped", "The lab shipped a revolutionary")
    revision = {"intro": "", "outro": "",
                "segments": [{"story": 1, "headline": "Lab ships agent", "key_fact": "3 steps", "text": GOOD_A},
                             {"story": 2, "headline": "x", "key_fact": "x", "text": "should be ignored"}]}
    llm = ScriptedLLM(_script(hype_a, GOOD_B), NO_ISSUES, revision, NO_ISSUES)
    ep = CriticWriter(llm, llm, "Show", "Host").write(_stories())
    assert [s.text for s in ep.story_segments] == [GOOD_A, GOOD_B]
    stages = [c[0] for c in llm.calls]
    assert stages == ["writer", "critic", "writer-repair-1", "critic"]
    assert "hype words" in llm.calls[2][1]


def test_critic_findings_that_survive_the_repairs_fall_back_per_segment():
    wrong_b = GOOD_B.replace("2 times", "5 times")
    flagged = {"segments": [{"story": 2, "unsupported": ["5 times faster"]}]}
    no_change = {"intro": "", "outro": "", "segments": []}
    llm = ScriptedLLM(_script(GOOD_A, wrong_b), flagged, no_change, flagged, no_change, flagged)
    ep = CriticWriter(llm, llm, "Show", "Host", max_repairs=2).write(_stories())
    assert ep.story_segments[0].text == GOOD_A  # untouched
    assert ep.story_segments[1].text.startswith("Chip is faster. The new chip is 2 times faster")  # from summary


def test_critic_writer_fills_a_missing_segment_instead_of_dropping_the_script():
    short = _script(GOOD_A, GOOD_B)
    short["segments"] = short["segments"][:1]
    llm = ScriptedLLM(short, NO_ISSUES)
    ep = CriticWriter(llm, llm, "Show", "Host", max_repairs=0).write(_stories())
    assert ep.story_segments[0].text == GOOD_A and ep.story_segments[1].text.startswith("Chip is faster.")
    assert ep.title == "AI today"


def test_lint_catches_numbers_urls_and_length_and_trim_fits():
    stories = _stories()
    ep = TemplateWriter("Show", "Host").write(stories)
    ep.segments[1].text = "Visit www.lab.ai to see the 7 step agent. " + " ".join(["word"] * 60)
    codes = {(i.code, i.index) for i in lint_episode(ep, stories)}
    assert {("url_in_narration", 0), ("unsupported_number", 0), ("too_long", 0)} <= codes
    long_ep = Episode("t", "d", [], [ep.segments[0]] + [Segment("story", " ".join(["word"] * 60))] * 8 +
                      [ep.segments[-1]])
    assert predicted_seconds(long_ep) > TARGET_MAX_SECONDS
    assert predicted_seconds(trim_to_fit(long_ep, TARGET_MAX_SECONDS)) <= TARGET_MAX_SECONDS


# --- rendering guards --------------------------------------------------------------------------

def test_captions_fit_the_frame():
    text = "Regulators outlined guidance for general-purpose accelerators delivering one-hundred-twenty-eight tokens"
    words = [Word(w, i * 0.4, i * 0.4 + 0.4) for i, w in enumerate(text.split())]
    for chunk in caption_chunks(words):
        line = " ".join(w.text for w in chunk)
        assert len(chunk) == 1 or caption_width(line) <= CAPTION_MAX_W
    assert "\\fs" in captions_ass(words)  # the one word too wide on its own is shrunk


def test_key_fact_chip_fits_the_card():
    f, text = chip_text("One hundred twenty-eight thousand token context window for every developer")
    assert f.getlength(text) <= CHIP_TEXT_MAX_W and text.endswith("…")
    f, text = chip_text("3x faster")
    assert text == "3x faster" and f.size == 40


def test_title_keeps_the_shorts_hashtag():
    ep = Episode("x" * 120, "", [], [])
    assert youtube_title(ep).endswith(" #Shorts") and len(youtube_title(ep)) <= 100


def test_qa_blocks_silent_segments_and_sample_stories(monkeypatch, tmp_path):
    monkeypatch.setattr(qa, "media_duration", lambda p: 120.0)
    monkeypatch.setattr(qa, "has_audio", lambda p: True)
    stories = [_story(f"S{i}", url=f"https://example.com/sample/{i}") for i in range(2)]
    ep = TemplateWriter("Show", "Host").write(stories)
    vo = Voiceover(tmp_path / "v.wav", 120.0, [], [], silent_segments=[2])
    report = qa.check(tmp_path / "v.mp4", ep, 2, vo)
    assert not report.passed
    assert any("story 2" in p for p in report.problems) and any("sample" in p for p in report.problems)
    assert qa.check(tmp_path / "v.mp4", ep, 2, Voiceover(tmp_path / "v.wav", 120.0, []), allow_sample=True).passed
    assert not qa.check(tmp_path / "v.mp4", ep, 3, None, allow_sample=True).passed  # too few stories


def test_narrate_reuses_unchanged_clips(tmp_path):
    spoken = []

    class CountingVoice(voice.SilentVoice):
        name = "counting"

        def speak(self, text, out_path):
            spoken.append(text)
            return super().speak(text, out_path)

    ep = TemplateWriter("Show", "Host").write(_stories())
    voice.narrate(CountingVoice(), ep, tmp_path)
    ep.segments[1].text = "A shorter first story."
    vo = voice.narrate(CountingVoice(), ep, tmp_path)
    assert len(spoken) == len(ep.segments) + 1 and spoken[-1] == "A shorter first story."
    assert vo.silent_segments == []


def test_reddit_reads_rss_and_keeps_outbound_links(monkeypatch):
    feed = """<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom">
    <entry><title>New open model beats GPT</title><link href="https://www.reddit.com/r/x/comments/1/"/>
      <updated>2026-09-26T10:00:00+00:00</updated>
      <content type="html">&lt;p&gt;Big release&lt;/p&gt; submitted by u/a &lt;a href="https://lab.ai/model"&gt;[link]&lt;/a&gt;</content></entry>
    <entry><title>Discussion thread</title><link href="https://www.reddit.com/r/x/comments/2/"/>
      <updated>2026-09-26T11:00:00+00:00</updated>
      <content type="html">&lt;a href="https://www.reddit.com/r/x/comments/2/"&gt;[link]&lt;/a&gt;</content></entry>
    </feed>"""
    urls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        urls.append(url)
        return SimpleNamespace(content=feed.encode(), raise_for_status=lambda: None)

    monkeypatch.setattr(sources.requests, "get", fake_get)
    monkeypatch.setattr(sources.time, "sleep", lambda s: None)
    monkeypatch.setattr(sources.RedditSource, "SUBS", ["x"])
    got = sources.RedditSource().fetch()
    assert urls == ["https://www.reddit.com/r/x/top/.rss"]
    assert [s.url for s in got] == ["https://lab.ai/model", "https://www.reddit.com/r/x/comments/2/"]
    assert got[0].summary == "Big release" and got[0].published.hour == 10
