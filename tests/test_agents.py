import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from shorts import pipeline, qa, sources, voice
from shorts.checks import (TARGET_MAX_SECONDS, check_picks, lint_episode, norm_url, predicted_seconds, same_event,
                           source_urls, unsupported_numbers)
from shorts.config import Config
from shorts.composer import CAPTION_MAX_W, caption_chunks, caption_width, captions_ass
from shorts.llm import PRICES, PRO_PRICE, UNKNOWN_PRICE, BudgetExceeded, OpenAILLM, SpendLedger, Usage, price
from shorts.models import Episode, Segment, Story, Voiceover, Word
from shorts.selection import AgentEditor, SeenStore, pick_with_fallback
from shorts.upload import youtube_title
from shorts.visuals import CHIP_TEXT_MAX_W, chip_text
from shorts.writer import CriticWriter, TemplateWriter, shorten, trim_to_fit

NOW = datetime.now(timezone.utc)


def _story(title, url=None, hours_ago=1, kind="article", summary="An AI model from OpenAI", body="",
           source="Feed"):
    return Story(title=title, url=url if url is not None else f"https://news.example/{title.replace(' ', '-')}",
                 source=source, published=NOW - timedelta(hours=hours_ago), summary=summary, kind=kind, body=body)


# Feed text with enough substance to air (description_problem needs 12+ words, 10+ beyond the headline).
SUMMARIES = {
    "OpenAI model launch": "OpenAI released a new reasoning model to ChatGPT users and developers. It scores higher on "
                           "coding and math tests and costs less to run than the model it replaces.",
    "OpenAI model": "OpenAI made its newest model the default in ChatGPT for free and paid users alike. The company "
                    "says it answers faster and makes fewer factual mistakes in long conversations.",
    "Nvidia AI chip": "Nvidia showed a data center chip built for serving AI models. The company says it cuts the "
                      "cost of each chatbot answer, and cloud providers get the first units this year.",
    "Nvidia unveils AI chip": "Nvidia unveiled a data center chip designed for running trained AI models. Cloud "
                              "providers say it will make chatbot answers cheaper to serve later this year.",
    "Google Gemini update": "Google rolled out an update to its Gemini assistant on Android phones. It can now read "
                            "what is on screen and take actions inside apps like Gmail and Maps when asked.",
    "OpenAI ships GPT agent": "OpenAI launched an agent inside ChatGPT that browses websites and fills in forms. "
                              "Paying users can ask it to book a table or order groceries, and it checks before buying.",
    "EU passes AI rules": "European lawmakers approved new rules for general purpose AI models. Developers of the "
                          "largest models must publish training data summaries and report serious incidents.",
    "Nvidia unveils inference chip": "Nvidia unveiled a chip made for running AI models rather than training them. "
                                     "Cloud companies expect it to lower what each chatbot answer costs them.",
    "EU passes AI audit rules": "The European Parliament voted for independent audits of high risk AI systems. "
                                "Companies selling hiring or credit scoring tools must prove they are tested for bias.",
    "Anthropic raises funding for Claude": "Anthropic raised new funding from investors led by a large tech fund. "
                                           "The money pays for computing power to train and serve future Claude models.",
    "Google Gemini tops math olympiad": "A Gemini model from Google solved most problems from this year's "
                                        "International Mathematical Olympiad, scoring at the level of top human students.",
    "Meta open sources Llama": "Meta released its latest Llama model with open weights for researchers and companies. "
                               "Developers can download it and run it on their own servers without paying fees.",
    "Mistral releases coding model": "French startup Mistral released a model that writes and fixes software code. "
                                     "It runs on a single graphics card and plugs into popular code editors.",
    "DeepMind robot learns to cook": "Google DeepMind trained a robot arm to prepare simple meals by watching videos "
                                     "of people cooking, then practicing each step in a simulated kitchen.",
}


def _news(title, **kw):
    return _story(title, summary=SUMMARIES[title], **kw)


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


def test_same_event_compares_names_and_numbers_not_headline_shape():
    assert not same_event("Nvidia unveils new AI chip", "AMD unveils new AI chip")
    assert not same_event("Anthropic raises 10 billion dollars", "OpenAI raises 40 billion dollars")
    assert same_event("OpenAI launches GPT-6 Sol", "OpenAI releases GPT-6 Sol model")
    assert same_event("Meta releases Llama 5", "Meta's Llama 5 is out")
    # the same number written differently is the same number
    assert same_event("Nvidia invests $100 billion in OpenAI", "Nvidia to invest up to $100B in OpenAI as part of deal")
    assert same_event("OpenAI launches GPT-5", "OpenAI launches GPT 5")
    assert same_event("Google releases Gemini 3", "Google launches Gemini 3.0")
    assert same_event("OpenAI launches GPT\u20116", "OpenAI launches GPT-6")  # typographic hyphen
    assert same_event("Anthropic raises $13B at $183B valuation",
                      "Anthropic raises $13 billion Series F, now valued at $183 billion")
    # a different partner, product or place is a different story
    assert not same_event("OpenAI signs chip deal with AMD", "OpenAI signs chip deal with Broadcom")
    assert not same_event("Google launches Gemini 3 Pro", "Google launches Gemini 3 Deep Think")
    assert not same_event("Anthropic opens office in Seoul", "Anthropic opens office in Tokyo")
    assert not same_event("Anthropic releases Claude Haiku 4.5", "Anthropic releases Claude Sonnet 4.5")
    picks = [_story("Nvidia unveils new AI chip"), _story("AMD unveils new AI chip"),
             _story("Anthropic raises 10 billion dollars"), _story("xAI raises $20B in Series E")]
    aired = ["OpenAI raises 40 billion dollars", "xAI raises $20 billion"]
    issues = [(i.code, i.index) for i in check_picks(picks, picks, 4, set(), aired, 30) if i.fatal]
    assert issues == [("already_aired", 3)]
    # words that start with a digit but aren't amounts ("2nm", "4o", "10x") don't break the check
    assert not same_event("TSMC starts 2nm chip production for Nvidia", "Apple books most of TSMC 2nm capacity for AI chips")
    assert not same_event("OpenAI retires GPT-4o voice in ChatGPT", "Microsoft moves Copilot off GPT-4o voice")
    assert not same_event("Nvidia Rubin is 10x faster at inference", "AMD claims 10x faster inference with MI500 chips")


def test_unparseable_links_in_newsletters_are_ignored():
    letter = _story("Letter", kind="newsletter", url="",
                    body="Run it at http://[::1]:8080 or unsubscribe: https://[UNSUBSCRIBE_LINK] (https://lab.ai/x)")
    assert source_urls([letter]) == {"https://lab.ai/x"}
    assert norm_url("https://[UNSUB") == ""


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
    candidates = [_news("OpenAI model launch"), _news("Nvidia AI chip"), _news("Google Gemini update")]
    reply = {"stories": [_pick("OpenAI model launch", "https://invented.link/x"),
                         _pick("Anthropic ships Claude agent", "https://news.example/other"),
                         _pick("OpenAI model launch again", candidates[0].url)]}
    picks = pick_with_fallback(AgentEditor(ScriptedLLM(reply, reply, reply), 30, max_repairs=2),
                               candidates, 3, seen, 30)
    assert len(picks) == 3
    assert picks[0].headline == "OpenAI model launch" and picks[0].url == ""  # made-up link removed, story kept
    assert "Anthropic ships Claude agent" not in [p.headline for p in picks]  # already aired
    assert {"Nvidia AI chip", "Google Gemini update"} <= {p.headline for p in picks}  # topped up


def test_editor_keeps_its_draft_when_a_repair_call_fails(tmp_path):
    candidates = [_news("OpenAI ships GPT agent"), _news("Nvidia unveils AI chip"), _news("EU passes AI rules")]
    draft = {"stories": [_pick("OpenAI ships GPT agent", candidates[0].url, summary="Written by the editor."),
                         _pick("Nvidia unveils AI chip", candidates[1].url),
                         _pick("OpenAI ships a GPT agent", candidates[0].url)]}
    llm = ScriptedLLM(draft, TimeoutError("read timed out"))
    picks = pick_with_fallback(AgentEditor(llm, 30), candidates, 3, SeenStore(tmp_path / "seen.json"), 30)
    assert [p.headline for p in picks] == ["OpenAI ships GPT agent", "Nvidia unveils AI chip", "EU passes AI rules"]
    assert picks[0].summary == "Written by the editor."  # the draft survived; only the duplicate was replaced


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


def test_restricted_key_falls_back_to_the_base_model():
    err = type("PermissionDeniedError", (Exception,), {"status_code": 403, "code": "model_not_found"})(
        "Project proj_x does not have access to model gpt-6-sol")
    base, fake = _openai(_resp(text='{"ok": true}'))
    llm = base.with_model("gpt-6-sol")
    fake.replies.insert(0, err)
    assert llm.json("s", "u", stage="editor") == {"ok": True} and llm.model == "gpt-5"


def test_prices_match_exact_names_and_dated_snapshots_only():
    assert price("gpt-5") == price("gpt-5-2025-08-07") == PRICES["gpt-5"]
    assert price("gpt-5-mini") == PRICES["gpt-5-mini"] and price("gpt-6-luna") == PRICES["gpt-6-luna"]
    assert price("gpt-5.5") == price("gpt-6-sol-mini") == UNKNOWN_PRICE
    assert price("gpt-5-pro") == price("gpt-6-sol-pro") == PRO_PRICE


def test_budget_cap_stops_calls_and_editor_falls_back(tmp_path):
    usage = Usage(run_cap_usd=0.01)
    usage.add("editor", "gpt-5", 100_000, 1000)  # $0.135, over the cap
    llm, fake = _openai(_resp(text="{}"), usage=usage)
    with pytest.raises(BudgetExceeded):
        llm.json("s", "u", stage="writer")
    assert fake.calls == []
    picks = pick_with_fallback(AgentEditor(llm, 30), [_news("OpenAI model"), _news("Nvidia AI chip")], 2,
                               SeenStore(tmp_path / "seen.json"), 30)
    assert len(picks) == 2


def test_spend_ledger_carries_the_month_across_runs(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.json")
    ledger.add(12.5)
    Usage(on_add=ledger.add).add("writer", "gpt-5", 1_000_000, 0)  # saved as it happens: +$1.25
    assert SpendLedger(tmp_path / "spend.json").this_month() == pytest.approx(13.75)
    ledger.months = {}
    ledger.add(12.5)
    usage = Usage(month_cap_usd=18, month_spent_usd=SpendLedger(tmp_path / "spend.json").this_month())
    usage.add("writer", "gpt-5", 5_000_000, 0)  # $6.25
    assert usage.over_budget()


# --- writer agent ----------------------------------------------------------------------------

def _stories():
    a = _story("Lab ships agent", summary="The lab shipped an agent that books travel in 3 steps. Users say where "
                                          "and when they want to go, the agent compares flights and hotels, and it "
                                          "asks for approval before paying.")
    b = _story("Chip is faster", summary="The new chip is 2 times faster at inference than the one it replaces. "
                                         "Cloud providers say faster inference makes chatbots cheaper to serve, and "
                                         "the first servers ship to customers this year.")
    for s in (a, b):
        s.headline = s.title
    return [a, b]


INTRO = "Quack, it's Host on Show! A faster chip is coming up. First, the lab's new agent."


def _script(a_text, b_text):
    return {"title": "AI today", "description": "Two stories.", "tags": ["ai"], "intro": INTRO,
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
    # from the summary, which restates the headline, so the headline isn't read out as well
    assert ep.story_segments[1].text.startswith("The new chip is 2 times faster")


def test_a_claim_only_the_critic_catches_falls_back_after_the_repairs():
    claim_a = GOOD_A.replace("asks before paying", "pays for everything itself")  # no number for the rules to see
    flagged = {"intro": [], "outro": [], "segments": [{"story": 1, "unsupported": ["pays for everything itself"]}]}
    no_change = {"intro": "", "outro": "", "segments": []}
    llm = ScriptedLLM(_script(claim_a, GOOD_B), flagged, no_change, flagged, no_change, flagged)
    ep = CriticWriter(llm, llm, "Show", "Host", max_repairs=2).write(_stories())
    assert not lint_episode(replace(ep, segments=[ep.segments[0], Segment("story", claim_a, "Lab ships agent"),
                                                  *ep.segments[2:]]), _stories())  # invisible to the rules
    assert ep.story_segments[0].text.startswith("The lab shipped")  # template line: the summary, headline once
    assert ep.story_segments[1].text == GOOD_B
    assert [c[0] for c in llm.calls].count("critic") == 3


def test_intro_and_outro_are_checked_and_fixed_without_touching_stories():
    script = _script(GOOD_A, GOOD_B)
    script["intro"] = "A revolutionary day: the lab raised 400 billion dollars, plus 1 more story."
    flagged = {"intro": [], "outro": ["claims the show is on every night"], "segments": []}
    revision = {"intro": "Quack, it's Host on Show! A chip that is 2 times faster is ahead. First, the lab's agent.",
                "outro": "",
                "segments": [{"story": 1, "headline": "x", "key_fact": "x", "text": "must be ignored"}]}
    llm = ScriptedLLM(script, flagged, revision, NO_ISSUES)
    ep = CriticWriter(llm, llm, "Show", "Host", max_repairs=1).write(_stories())
    problems = llm.calls[2][1]
    assert "the intro uses hype words" in problems and "400 billion" in problems
    assert "the outro" not in problems  # the outro is the show's own: not the critic's to fix
    assert ep.segments[0].text == revision["intro"]
    assert [s.text for s in ep.story_segments] == [GOOD_A, GOOD_B]  # frame problems don't open the stories
    llm = ScriptedLLM(script, NO_ISSUES)
    ep = CriticWriter(llm, llm, "Show", "Host", max_repairs=0).write(_stories())
    standard = TemplateWriter("Show", "Host").intro_for(_stories(), ep.story_segments).text
    assert ep.segments[0].text == standard  # still bad: standard intro
    assert standard.startswith(("Quack", "Waddle", "Ruffle", "Fresh", "Splash")) and "Chip is faster" in standard
    no_change = {"intro": "", "outro": "", "segments": []}
    llm = ScriptedLLM(_script(GOOD_A, GOOD_B), flagged, no_change, flagged)
    ep = CriticWriter(llm, llm, "Show", "Host", max_repairs=1).write(_stories())
    assert ep.segments[-1].text == TemplateWriter("Show", "Host").outro().text  # always the show's outro
    assert [c[0] for c in llm.calls] == ["writer", "critic"]  # a finding on the fixed outro costs no rewrite
    assert [s.text for s in ep.story_segments] == [GOOD_A, GOOD_B]


def test_critic_sees_the_whole_article_and_the_show_details():
    stories = _stories()
    stories[0].body = "x " * 1600 + "MARKER fact"
    script = _script(GOOD_A, GOOD_B)
    script["title"] = "OpenAI raises 400 billion dollars"  # invented number in the YouTube title
    llm = ScriptedLLM(script, NO_ISSUES)
    ep = CriticWriter(llm, llm, "Duck Desk", "Quackers", max_repairs=0).write(stories)
    critic_prompt = llm.calls[1][1]
    assert "MARKER fact" in critic_prompt and "(Duck Desk)" in critic_prompt and "(Quackers)" in critic_prompt
    assert ep.title == TemplateWriter("Duck Desk", "Quackers").write(stories).title


def test_trim_rewrite_is_checked_like_the_draft():
    stories = _stories()
    llm = ScriptedLLM(_script(GOOD_A, GOOD_B), NO_ISSUES)
    writer = CriticWriter(llm, llm, "Show", "Host", max_repairs=0)
    ep = writer.write(stories)
    short_a = "A revolutionary lab agent books travel in 9 steps."  # shorter, but hype and a new number
    short_b = "A new chip runs AI models 2 times faster at inference, so serving gets cheaper."
    llm.replies = [{"intro": "A revolutionary day with 2 stories.", "outro": "Bye.", "segments": [
        {"story": 1, "headline": "Renamed", "key_fact": "9 steps", "text": short_a},
        {"story": 2, "headline": "Renamed too", "key_fact": "", "text": short_b}]}, NO_ISSUES]
    trimmed = writer.shorten(ep, stories, 10)
    assert [s.text for s in trimmed.story_segments] == [GOOD_A, short_b]  # the broken trim kept its old text
    assert [s.headline for s in trimmed.story_segments] == ["Lab ships agent", "Chip is faster"]
    assert trimmed.segments[0] == ep.segments[0] and trimmed.segments[-1] == ep.segments[-1]
    assert [c[0] for c in llm.calls][-2:] == ["writer-trim", "critic"]


def test_length_trim_paces_from_the_script_that_was_voiced():
    long_text = " ".join(["word"] * 60) + "."
    stories = [_story(f"S{i}") for i in range(8)]
    ep = TemplateWriter("Show", "Host").write(stories)
    ep = replace(ep, segments=[ep.segments[0], *[Segment("story", long_text)] * 8, ep.segments[-1]])
    cut = replace(ep, segments=[ep.segments[0], *[Segment("story", " ".join(["word"] * 40) + ".")] * 8,
                                ep.segments[-1]])

    class FixedTrim(CriticWriter):
        def shorten(self, episode, stories, seconds_over):
            return cut

    writer = FixedTrim(None, None, "Show", "Host")
    measured = predicted_seconds(ep)  # the voice ran exactly at the predicted pace
    assert predicted_seconds(cut) < TARGET_MAX_SECONDS < measured
    assert shorten(writer, ep, stories, measured) == cut  # already fits: nothing more is cut


def test_critic_writer_fills_a_missing_segment_instead_of_dropping_the_script():
    short = _script(GOOD_A, GOOD_B)
    short["segments"] = short["segments"][:1]
    llm = ScriptedLLM(short, NO_ISSUES)
    ep = CriticWriter(llm, llm, "Show", "Host", max_repairs=0).write(_stories())
    assert ep.story_segments[0].text == GOOD_A and ep.story_segments[1].text.startswith("The new chip is 2 times")
    assert ep.title == "AI today"


def test_number_rule_allows_rounding_and_spelled_out_numbers_but_not_new_ones():
    fine = [("93 percent", "93.4%"), ("about 1.5 billion dollars", "$1.49 billion"), ("40 thousand", "40,000"),
            ("1.5 million", "$1,500,000"), ("3 new models", "three new models"), ("GPT 4 point 1", "GPT-4.1"),
            ("20 percent", "19.6%"), ("1.5b", "$1.49 billion")]
    for script, material in fine:
        assert unsupported_numbers(script, material) == [], (script, material)
    assert unsupported_numbers("1.4 trillion dollars", "OpenAI signed $1.4T in compute deals") == []
    assert unsupported_numbers("a 1 million token context", "a million-token context window") == []
    assert unsupported_numbers("1 billion users", "one billion users") == []
    # a wrong model version is not a rounding
    assert unsupported_numbers("GPT-6 is now in ChatGPT", "OpenAI released GPT-5.5") == ["6"]
    assert unsupported_numbers("Claude Sonnet 5", "Claude Sonnet 4.5") == ["5"]
    assert unsupported_numbers("in 2026", "in 2025") == ["2026"]
    assert unsupported_numbers("900 million dollars", "40 billion") == ["900 million"]
    assert unsupported_numbers("5 times faster", "faster") == ["5"]
    story = _story("Report", summary="A report says models got faster.", source="404 Media")
    story.headline = story.title
    ep = TemplateWriter("Show", "Host").write([story])
    ep.segments[1].text = "According to 404 Media, a report says models got faster."
    assert not [i for i in lint_episode(ep, [story]) if i.code == "unsupported_number"]
    ep.segments[1].headline = "Models got 900 percent faster"  # the big on-screen headline is checked too
    assert [i.code for i in lint_episode(ep, [story]) if i.code == "unsupported_number"] == ["unsupported_number"]


def test_intro_may_use_the_show_name_and_date():
    story = _story("Report", summary="A report says models got faster.")
    ep = TemplateWriter("AI in 60 Seconds", "Quackers").write([story])
    ep.segments[0].text = "Happy Friday, September 26! It's AI in 60 Seconds, with 1 story today."
    frame = "AI in 60 Seconds Quackers Friday, September 26, 2026"
    assert [i.code for i in lint_episode(ep, [story]) if i.code == "intro_problem"] == ["intro_problem"]
    assert not [i for i in lint_episode(ep, [story], frame) if i.code == "intro_problem"]
    script = _script(GOOD_A, GOOD_B)
    script["intro"] = "It's AI in 60 Seconds with Quackers, and 2 AI stories today."
    llm = ScriptedLLM(script, NO_ISSUES)
    ep = CriticWriter(llm, llm, "AI in 60 Seconds", "Quackers").write(_stories())
    assert [c[0] for c in llm.calls] == ["writer", "critic"] and ep.segments[0].text == script["intro"]


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


def test_live_run_with_no_voice_fails_qa(monkeypatch, tmp_path):
    monkeypatch.setenv("SHORTS_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("SHORTS_STATE_DIR", str(tmp_path / "state"))
    titles = ["OpenAI ships GPT agent", "Nvidia unveils inference chip", "EU passes AI audit rules",
              "Anthropic raises funding for Claude", "Google Gemini tops math olympiad", "Meta open sources Llama",
              "Mistral releases coding model", "DeepMind robot learns to cook"]
    live = [_news(t) for t in titles]
    monkeypatch.setattr(pipeline, "fetch_all", lambda sources: live)
    monkeypatch.setattr(pipeline.composer, "render", lambda vo, cards, desk, host, out, preset: out)
    monkeypatch.setattr(qa, "media_duration", lambda p: 120.0)
    monkeypatch.setattr(qa, "has_audio", lambda p: True)
    # live news and the template writer (SHORTS_ALLOW_NO_AI=on, or there is no episode without a model),
    # but SHORTS_VOICE=silent
    cfg = replace(Config.from_env().offline(), sources=["rss"], allow_no_ai=True)
    with pytest.raises(RuntimeError, match="no voice at all"):
        pipeline.run(cfg)
    assert not (tmp_path / "state" / "seen_urls.json").exists()
    assert not (tmp_path / "state" / "intros.json").exists()


def test_live_run_without_a_model_stops_before_fetching(monkeypatch, tmp_path):
    monkeypatch.setenv("SHORTS_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("SHORTS_STATE_DIR", str(tmp_path / "state"))
    fetched = []
    monkeypatch.setattr(pipeline, "fetch_all", lambda sources: fetched.append(sources) or [])
    cfg = replace(Config.from_env().offline(), sources=["rss"], allow_no_ai=False)
    with pytest.raises(RuntimeError, match="No AI model is set up"):
        pipeline.run(cfg)
    assert fetched == []


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
