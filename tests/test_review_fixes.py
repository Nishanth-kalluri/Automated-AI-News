import json
import os
import re
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import requests

import shorts.writer as writer_mod
from shorts import pipeline, sources
from shorts.checks import Issue, lint_episode
from shorts.config import DEFAULT_OUTRO, Config
from shorts.content import description_problem, script_problems
from shorts.llm import BudgetExceeded
from shorts.models import Episode, Segment, Story
from shorts.selection import (EDITOR_SYSTEM, AgentEditor, LLMEditor, SeenStore, drop_duplicates, pick_stories,
                              pick_with_fallback, settle)
from shorts.writer import (TEMPLATE_HOOKS, WRITER_SYSTEM, CriticWriter, IntroLog, LLMWriter, TemplateWriter,
                           WriterFailed, _story_block, build_writer, default_description, episode_from_json,
                           episode_json, template_segment, write_episode)

NOW = datetime.now(timezone.utc)
SIGN_OFF = "That's the news from the pond. See you tomorrow!"
MIT_FEED = "https://www.technologyreview.com/topic/artificial-intelligence/feed"
LAB_FEED = "https://example-labs.dev/feed.xml"
INBOX = "news@agentmail.to"

# Feed text with enough substance to air (12+ words, 10+ beyond the headline).
SUMMARIES = {
    "OpenAI ships GPT agent": "OpenAI launched an agent inside ChatGPT that browses websites and fills in forms. "
                              "Paying users can ask it to book a table or order groceries, and it checks before buying.",
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
}
GOOD_TITLES = list(SUMMARIES)


def _story(title, url=None, hours_ago=1, kind="article", summary="An AI model from OpenAI", body="",
           source="Feed", popularity=0.0):
    return Story(title=title, url=url if url is not None else f"https://news.example/{title.replace(' ', '-')}",
                 source=source, published=NOW - timedelta(hours=hours_ago), summary=summary, kind=kind, body=body,
                 popularity=popularity)


def _news(title, **kw):
    return _story(title, summary=SUMMARIES[title], **kw)


def _as_pick(story):
    return replace(story, headline=story.title, outlets=[story.source])


def _pick(headline, url, summary, outlets=("TLDR AI",)):
    return {"headline": headline, "summary": summary, "key_fact": "", "url": url, "outlets": list(outlets), "why": ""}


class ScriptedLLM:
    """Replies in order; records every call. No run_tools, like the Anthropic client."""

    name, model = "scripted", "scripted-1"

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def json(self, system, user, *, stage, schema=None):
        self.calls.append(SimpleNamespace(stage=stage, system=system, user=user, schema=schema))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    @property
    def stages(self):
        return [c.stage for c in self.calls]


class FakeHTTP:
    """A requests.Response stand-in: status_code, text, json() and a raise_for_status that raises
    requests.HTTPError carrying the response, like the real one."""

    def __init__(self, data=None, status=200, content=b""):
        self.data, self.status_code, self.content = data if data is not None else {}, status, content
        self.text = json.dumps(self.data) if status < 400 else f'{{"error": "status {status}"}}'

    def json(self):
        return self.data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error", response=self)


def _clean_env(monkeypatch, tmp_path=None):
    for key in list(os.environ):
        if key.startswith("SHORTS_") or key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AGENTMAIL_API_KEY",
                                                 "AGENTMAIL_INBOX", "TAVILY_API_KEY"):
            monkeypatch.delenv(key, raising=False)
    if tmp_path is not None:
        monkeypatch.setenv("SHORTS_OUTPUT_DIR", str(tmp_path / "out"))
        monkeypatch.setenv("SHORTS_STATE_DIR", str(tmp_path / "state"))


def _first_words(text, n=4):
    return " ".join(re.findall(r"[a-z0-9']+", text.lower())[:n])


# --- sources -----------------------------------------------------------------------------------

MIT_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<title>Artificial intelligence – MIT Technology Review</title>
<link>https://www.technologyreview.com</link>
<item>
  <title>OpenAI&#8217;s screenless device listens all day</title>
  <link>https://www.technologyreview.com/2026/09/28/1124012/openai-screenless-device/</link>
  <pubDate>Mon, 28 Sep 2026 10:00:00 +0000</pubDate>
  <description><![CDATA[<p>This story originally appeared in The Algorithm, our weekly newsletter on AI. To get stories like this in your inbox first, <a href="https://forms.technologyreview.com/newsletters/">sign up here</a>.</p>
  <p>OpenAI is building a pocket device with no screen that listens to conversations all day. The company plans to ship it to developers next year.</p>]]></description>
</item>
</channel></rss>"""

LAB_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<title>AI News | Example Labs Blog</title>
<item>
  <title>Example Labs opens its robot model to researchers</title>
  <link>https://example-labs.dev/news/robot-model</link>
  <pubDate>Mon, 28 Sep 2026 12:00:00 +0000</pubDate>
  <description><![CDATA[<p>Example Labs released the weights of its robot control model for academic use. Labs can fine tune it on their own arms.</p>
  <p>The post <a href="https://example-labs.dev/news/robot-model">Example Labs opens its robot model</a> appeared first on <a href="https://example-labs.dev">Example Labs Blog</a>.</p>
  <p>Read more at our site. 412 points and 88 comments on HN.</p>]]></description>
</item>
</channel></rss>"""


def test_rss_drops_publication_boilerplate_and_credits_the_publisher(monkeypatch):
    feeds = {MIT_FEED: MIT_RSS, LAB_FEED: LAB_RSS}
    monkeypatch.setattr(sources.requests, "get",
                        lambda url, headers=None, timeout=None: FakeHTTP(content=feeds[url].encode()))
    mit, lab = sources.RSSSource([MIT_FEED, LAB_FEED]).fetch()
    assert mit.source == "MIT Technology Review"  # not "Artificial intelligence – MIT Technology Review"
    assert lab.source == "Example Labs Blog"  # the publisher part of the feed title
    assert mit.summary == ("OpenAI is building a pocket device with no screen that listens to conversations all day. "
                           "The company plans to ship it to developers next year.")
    assert lab.summary == ("Example Labs released the weights of its robot control model for academic use. "
                           "Labs can fine tune it on their own arms.")
    for story in (mit, lab):
        low = story.summary.lower()
        for said in ("newsletter", "originally appeared", "sign up", "inbox", "appeared first", "points",
                     "comments", "on hn", "read more"):
            assert said not in low, (said, story.summary)
        assert script_problems(story.summary) == []
        assert not description_problem(story)  # what's left still describes the news


def test_hacker_news_stories_carry_no_points_or_comments_text(monkeypatch):
    hits = [{"title": "Lab releases open robot model", "url": "https://lab.ai/robot", "objectID": "1",
             "created_at_i": int(NOW.timestamp()) - 3600, "points": 412, "num_comments": 88},
            {"title": "Ask HN: Is the new model any good?", "url": None, "objectID": "2",
             "created_at_i": int(NOW.timestamp()) - 7200, "points": 120, "num_comments": 300}]
    monkeypatch.setattr(sources.HackerNewsSource, "QUERIES", ["AI"])
    monkeypatch.setattr(sources, "hn_search", lambda query, since, min_points=0, hits_=30: hits)
    got = sources.HackerNewsSource(30).fetch()
    assert [s.summary for s in got] == ["", ""]
    assert got[0].popularity == pytest.approx(412 / 500)  # points only rank the story
    assert got[1].url == "https://news.ycombinator.com/item?id=2"
    # neither can air on its own: no description, and a forum thread is not news
    assert description_problem(got[0]) and description_problem(got[1])


def test_config_defaults_leave_forums_out_and_set_the_review_fixes(monkeypatch):
    _clean_env(monkeypatch)
    cfg = Config.from_env()
    assert cfg.sources == ["newsletter", "rss"]
    assert [s.name for s in sources.build_sources(cfg)] == ["newsletter", "rss"]
    assert cfg.min_stories == 4 and cfg.allow_no_ai is False
    assert cfg.outro == DEFAULT_OUTRO and cfg.outro.endswith(SIGN_OFF) and "subscribe" in cfg.outro.lower()
    assert cfg.edge_rate == "+18%"  # a faster voice
    monkeypatch.setenv("SHORTS_SOURCES", "rss,hackernews")
    monkeypatch.setenv("SHORTS_ALLOW_NO_AI", "on")
    monkeypatch.setenv("SHORTS_MIN_STORIES", "5")
    monkeypatch.setenv("SHORTS_VOICE_LINEUP", "edge:en-US-AnaNeural, openai:coral")
    cfg = Config.from_env()
    assert cfg.sources == ["rss", "hackernews"] and cfg.allow_no_ai and cfg.min_stories == 5
    assert cfg.voice_lineup == ["edge:en-US-AnaNeural", "openai:coral"]
    assert cfg.offline().voice_lineup == []


def _newsletter_http(calls, listings, messages):
    """A fake AgentMail: ``listings`` answers the /messages calls in order, ``messages`` by id."""

    def fake_get(url, params=None, headers=None, timeout=None):
        assert headers["Authorization"] == "Bearer key"
        if url.endswith("/messages"):
            calls.append(("list", dict(params or {})))
            return listings.pop(0)
        mid = url.rsplit("/", 1)[1]
        calls.append(("get", mid))
        return FakeHTTP(messages[mid])

    return fake_get


def _ts(hours_ago):
    return f"{NOW - timedelta(hours=hours_ago):%Y-%m-%dT%H:%M:%SZ}"


def test_newsletter_listing_retries_without_after_on_400_and_drops_old_mail(monkeypatch):
    long_html = "<p>" + "OpenAI shipped a new model to developers today. " * 12 + "</p>"
    listing = {"messages": [{"message_id": "new", "timestamp": _ts(2)},
                            {"message_id": "old", "timestamp": _ts(80)},
                            {"message_id": "undated"}]}
    messages = {"new": {"subject": "Today in AI", "from": "Daily Dose <hi@dose.ai>", "timestamp": _ts(2),
                        "html": long_html},
                "old": {"subject": "Last week", "from": "Daily Dose <hi@dose.ai>", "timestamp": _ts(80),
                        "html": long_html},
                "undated": {"subject": "Three days ago", "from": "Daily Dose <hi@dose.ai>", "timestamp": _ts(72),
                            "html": long_html}}
    calls = []
    monkeypatch.setattr(sources.requests, "get",
                        _newsletter_http(calls, [FakeHTTP(status=400), FakeHTTP(listing)], messages))
    got = sources.NewsletterSource("key", INBOX, 30).fetch()
    lists = [params for kind, params in calls if kind == "list"]
    assert len(lists) == 2 and lists[1] == {"limit": 50}
    assert lists[0]["limit"] == 50 and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", lists[0]["after"])
    fetched = [mid for kind, mid in calls if kind == "get"]
    assert "old" not in fetched  # too old by the listing's date: never downloaded
    assert fetched == ["new", "undated"]
    assert [s.title for s in got] == ["Today in AI"]  # the undated one is too old by its own date
    assert got[0].kind == "newsletter" and got[0].source == "Daily Dose"


def test_newsletter_listing_gives_up_after_the_plain_listing_is_rejected(monkeypatch):
    calls = []
    monkeypatch.setattr(sources.requests, "get",
                        _newsletter_http(calls, [FakeHTTP(status=400) for _ in range(3)], {}))
    assert sources.NewsletterSource("key", INBOX, 30).fetch() == []
    assert [params for _, params in calls][1:] == [{"limit": 50}, {}]


@pytest.mark.parametrize("status", [401, 500])
def test_newsletter_listing_does_not_retry_other_errors(monkeypatch, status):
    calls = []
    monkeypatch.setattr(sources.requests, "get", _newsletter_http(calls, [FakeHTTP(status=status)], {}))
    assert sources.NewsletterSource("key", INBOX, 30).fetch() == []
    assert len(calls) == 1


# --- selection ---------------------------------------------------------------------------------

def test_editor_is_asked_for_up_to_n_solid_stories_and_no_forum_talk(tmp_path):
    seen = SeenStore(tmp_path / "seen.json")
    letter = _story("Daily Dose", url="", kind="newsletter", body="OpenAI shipped an agent.")
    feed = [_news(t) for t in GOOD_TITLES[:4]]
    prompt = LLMEditor(ScriptedLLM(), 30)._prompt([letter, *feed], 6, seen)
    assert "Pick up to 6 stories" in prompt and "fewer if there aren't 6" in prompt
    assert "exactly" not in prompt
    assert "up to the requested number" in EDITOR_SYSTEM and "exactly the requested" not in EDITOR_SYSTEM
    assert "Skip Hacker News and Reddit threads" in EDITOR_SYSTEM
    assert "Never mention newsletters or Hacker News" in EDITOR_SYSTEM and "never give points" in EDITOR_SYSTEM
    assert "News about Reddit the company is fine" in EDITOR_SYSTEM


def test_agent_editor_accepts_fewer_picks_than_n_when_at_least_the_minimum(tmp_path):
    candidates = [_news(t) for t in GOOD_TITLES]
    reply = {"stories": [_pick(c.title, c.url, c.summary) for c in candidates[:4]]}
    llm = ScriptedLLM(reply)
    picks = AgentEditor(llm, 30, min_stories=4).pick(candidates, 8, SeenStore(tmp_path / "seen.json"))
    assert [p.headline for p in picks] == GOOD_TITLES[:4]
    assert llm.stages == ["editor"]  # 4 of 8 is fine: no repair round asking for padding


def test_to_stories_credits_the_publisher_never_the_newsletter():
    feed = _news("Meta open sources Llama", url="https://techcrunch.com/2026/09/28/meta-llama/", source="TechCrunch")
    summary = SUMMARIES["OpenAI ships GPT agent"]
    data = {"stories": [
        _pick("OpenAI ships GPT agent", "https://www.theverge.com/2026/9/28/openai-agent", summary),
        _pick("EU passes AI audit rules", "", SUMMARIES["EU passes AI audit rules"]),
        _pick("Anthropic raises funding", "https://tldr.tech/ai/2026-09-28", SUMMARIES["Anthropic raises funding for Claude"]),
        _pick("Meta open sources Llama", feed.url, feed.summary, outlets=["The Neuron"]),
    ]}
    stories = LLMEditor.to_stories(data, [feed], 4)
    assert [s.source for s in stories] == ["The Verge", "", "", "TechCrunch"]
    assert stories[0].outlets == ["TLDR AI"]  # the newsletter stays in outlets, off the credit
    assert stories[3].outlets == ["The Neuron", "TechCrunch"]


def test_settle_clears_a_credit_that_came_from_a_made_up_link(tmp_path):
    candidates = [_news("OpenAI ships GPT agent")]
    made_up = _story("Nvidia unveils inference chip", url="https://techcrunch.com/2026/09/28/invented/",
                     summary=SUMMARIES["Nvidia unveils inference chip"], source="TechCrunch")
    other = _story("EU passes AI audit rules", url="https://www.reuters.com/invented", source="Axios",
                   summary=SUMMARIES["EU passes AI audit rules"])
    for s in (made_up, other):
        s.headline = s.title
    kept = settle([made_up, other], candidates, 8, SeenStore(tmp_path / "seen.json"), 30)
    assert kept == [made_up, other]  # the stories stay
    assert (made_up.url, made_up.source) == ("", "")  # "TechCrunch" came only from the invented link
    assert (other.url, other.source) == ("", "Axios")  # a credit from elsewhere is kept


def _fallback_candidates():
    good = [_news(t, hours_ago=i + 1) for i, t in enumerate(GOOD_TITLES)]
    # fresher and more "AI" than the good ones, so the keyword ranking puts them first
    thin = _story("Microsoft Copilot gets AI agents", summary="Microsoft Copilot gets AI agents.", hours_ago=0.1,
                  popularity=1.0)
    no_text = _story("Show HN: an LLM inference engine in Rust", url="https://github.com/someone/engine",
                     summary="", source="Hacker News", hours_ago=0.1, popularity=1.0)
    forum = _story("Ask HN: Is the new OpenAI AI model worth it?", url="https://news.ycombinator.com/item?id=4242",
                   summary="People who tried the new OpenAI model compare it with the old one for coding, writing "
                           "and research, and most say the AI agent features are the biggest change.",
                   source="Hacker News", hours_ago=0.1, popularity=1.0)
    return good, [thin, no_text, forum]


class FixedEditor:
    name = "fixed"

    def __init__(self, picks):
        self.picks = picks

    def pick(self, candidates, n, seen):
        if isinstance(self.picks, Exception):
            raise self.picks
        return [_as_pick(s) for s in self.picks]


def test_fallback_does_not_pad_when_the_editor_gave_at_least_the_minimum(tmp_path):
    good, weak = _fallback_candidates()
    picks = pick_with_fallback(FixedEditor(good[:4]), good + weak, 8, SeenStore(tmp_path / "seen.json"), 30,
                               min_n=4)
    assert [p.headline for p in picks] == GOOD_TITLES[:4]


def test_fallback_tops_up_to_the_minimum_only_with_described_news(tmp_path):
    good, weak = _fallback_candidates()
    weak_titles = {s.title for s in weak}
    picks = pick_with_fallback(FixedEditor(good[:2]), good + weak, 8, SeenStore(tmp_path / "seen.json"), 30,
                               min_n=4)
    assert len(picks) == 4 and [p.headline for p in picks[:2]] == GOOD_TITLES[:2]
    assert not {p.title for p in picks} & weak_titles
    # the editor failed outright: up to n, still never a thin story or a forum thread
    picks = pick_with_fallback(FixedEditor(RuntimeError("down")), good + weak, 8,
                               SeenStore(tmp_path / "seen2.json"), 30, min_n=4)
    assert sorted(p.title for p in picks) == sorted(GOOD_TITLES)
    assert not any(description_problem(p) for p in picks)


def test_pick_stories_keeps_one_story_per_event_across_outlets():
    stories = [
        _story("Nvidia launches new platform for reining in rogue AI agents", source="TechCrunch",
               url="https://techcrunch.com/2026/09/28/nvidia-rogue-agents/"),
        _story("Nvidia says its new AI safety platform can contain rogue agents within 'milliseconds'",
               source="The Verge", url="https://www.theverge.com/news/nvidia-safety-platform", hours_ago=2),
        _news("Meta open sources Llama", hours_ago=3),
        _story("Ask HN: Nvidia's rogue agent platform", url="https://news.ycombinator.com/item?id=1", hours_ago=0.5),
    ]
    picked = pick_stories(stories, 5, 30)
    titles = [s.title for s in picked]
    assert len(picked) == 2 and sum(t.startswith("Nvidia") for t in titles) == 1
    assert "Meta open sources Llama" in titles


def _three():
    stories = [_news(t) for t in ("OpenAI ships GPT agent", "Nvidia unveils inference chip", "EU passes AI audit rules")]
    for s in stories:
        s.headline = s.title
    return stories


def test_drop_duplicates_drops_the_later_story_of_the_same_event():
    stories = _three()
    llm = ScriptedLLM({"duplicates": [{"story": 3, "same_as": 1}]})
    assert drop_duplicates(llm, stories) == stories[:2]
    call = llm.calls[0]
    assert call.stage == "dedupe" and "1. OpenAI ships GPT agent" in call.user and "3. EU passes" in call.user


def test_drop_duplicates_ignores_numbers_that_make_no_sense():
    stories = _three()
    nonsense = {"duplicates": [{"story": 1, "same_as": 2}, {"story": 4, "same_as": 1}, {"story": 2, "same_as": 0},
                               {"story": 2, "same_as": 2}, {"story": -1, "same_as": -3}]}
    assert drop_duplicates(ScriptedLLM(nonsense), stories) == stories


@pytest.mark.parametrize("error", [RuntimeError("model down"), BudgetExceeded("over budget")])
def test_drop_duplicates_keeps_the_picks_when_the_model_fails(error):
    stories = _three()
    assert drop_duplicates(ScriptedLLM(error), stories) == stories
    assert drop_duplicates(None, stories) == stories
    assert drop_duplicates(ScriptedLLM(), stories[:1]) == stories[:1]  # nothing to compare: no call


def test_drop_duplicates_never_raises_on_a_malformed_reply():
    stories = _three()
    assert drop_duplicates(ScriptedLLM({"duplicates": [{"story": "3", "same_as": "1"}]}), stories) in (
        stories, stories[:2])


# --- writer ------------------------------------------------------------------------------------

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


def _lazy_story():
    s = _story("GPT-6 feels lazy, Reddit users say",
               summary="Reddit users say GPT-6 gives shorter answers than GPT-5 did. OpenAI says it is looking into "
                       "the reports and will share an update soon.")
    s.headline = s.title
    return s


GOOD_A = "The lab shipped an agent that books your travel in 3 steps. You say where and when, it compares options " \
         "and asks before paying. Why it matters: assistants are starting to finish whole errands for you."
GOOD_B = "A new chip runs AI models 2 times faster at inference. Serving chatbots gets cheaper, and cheaper serving " \
         "usually means more free features for users soon. Watch for the big clouds to adopt it first this year."
BETTER_B = "The new chip does inference 2 times faster than the one it replaces. Cloud providers expect cheaper " \
           "chatbot answers, and the first servers reach customers this year. Cheaper serving means more for users."
NO_ISSUES = {"intro": [], "outro": [], "segments": []}
NO_CHANGE = {"intro": "", "outro": "", "segments": []}


def _script(*texts, intro="Hello pond, a lab agent books trips and a chip gets faster."):
    heads = [("Lab ships agent", "3 steps"), ("Chip is faster", "2x faster"), ("GPT-6 feels lazy", "")]
    return {"title": "AI today", "description": "Today's stories.", "tags": ["ai"], "intro": intro,
            "segments": [{"headline": h, "key_fact": k, "text": t} for (h, k), t in zip(heads, texts)],
            "outro": ""}


def _on_day(monkeypatch, day):
    monkeypatch.setattr(writer_mod, "date", SimpleNamespace(today=lambda: day))


def test_writer_rules_ban_source_talk_and_ask_for_a_fresh_hook():
    for rule in ("Never mention newsletters or Hacker News", "never read out points, upvotes or comment counts",
                 "never ask viewers to subscribe", "never say the\n  show's sign-off",
                 "fresh, playful hook", "never open like one of the recent intros", "Don't count the stories",
                 '"outro": always ""', "never say the same thing twice", "our newsletter"):
        assert rule in WRITER_SYSTEM, rule
    llm = ScriptedLLM(_script(GOOD_A, GOOD_B))
    LLMWriter(llm, "Duck Desk", "Quackers").write(_stories())
    assert "Hacker News" in llm.calls[0].system and "{n}" not in llm.calls[0].system


def test_draft_prompt_lists_the_recent_intros_to_avoid():
    recent = ["Quack quack, it's Quackers on Duck Desk! Big day.", "Waddle in, friends, OpenAI did a thing."]
    llm = ScriptedLLM(_script(GOOD_A, GOOD_B))
    build_writer(llm, "Duck Desk", "Quackers", agents=False, recent_intros=recent).write(_stories())
    user = llm.calls[0].user
    assert "Recent intros (open differently from all of these):" in user
    assert all(f"- {intro}\n" in user for intro in recent)
    llm = ScriptedLLM(_script(GOOD_A, GOOD_B))
    LLMWriter(llm, "Duck Desk", "Quackers").write(_stories())
    assert "Recent intros" not in llm.calls[0].user


def test_intro_opening_like_a_recent_episode_goes_back_for_a_fresh_hook():
    recent = ["Quack quack, it's Quackers on Duck Desk! Big news from OpenAI today."]
    stale = _script(GOOD_A, GOOD_B, intro="Quack quack, it's Quackers on Duck Desk! A lab agent now books trips.")
    fresh = "Pack your bags: a lab agent now books whole trips for you, and a chip gets faster."
    llm = ScriptedLLM(stale, NO_ISSUES, dict(NO_CHANGE, intro=fresh), NO_ISSUES)
    ep = CriticWriter(llm, llm, "Duck Desk", "Quackers", max_repairs=1, recent_intros=recent).write(_stories())
    assert llm.stages == ["writer", "critic", "writer-repair-1", "critic"]
    assert "like a recent episode" in llm.calls[2].user
    assert ep.segments[0].text == fresh


def test_the_outro_is_always_the_fixed_one():
    script = _script(GOOD_A, GOOD_B)
    script["outro"] = "Follow us for more AI news every day!"
    ep = LLMWriter(ScriptedLLM(script), "Duck Desk", "Quackers").write(_stories())
    assert ep.segments[-1] == Segment("outro", DEFAULT_OUTRO)
    custom = f"Subscribe for tomorrow's pond report. {SIGN_OFF}"
    ep = LLMWriter(ScriptedLLM(script), "Duck Desk", "Quackers", outro=custom).write(_stories())
    assert ep.segments[-1].text == custom
    assert TemplateWriter("Duck Desk", "Quackers", outro=custom).write(_stories()).segments[-1].text == custom
    assert build_writer(None, "Duck Desk", "Quackers", outro=custom).write(_stories()).segments[-1].text == custom


def test_the_critic_writer_keeps_the_fixed_outro_through_revisions():
    hype_a = GOOD_A.replace("The lab shipped", "The lab shipped a revolutionary")
    script = _script(hype_a, GOOD_B)
    script["outro"] = "Smash that like button!"
    revision = {"intro": "", "outro": "Bye bye, and follow for more!",
                "segments": [{"story": 1, "headline": "Lab ships agent", "key_fact": "3 steps", "text": GOOD_A}]}
    llm = ScriptedLLM(script, NO_ISSUES, revision, NO_ISSUES)
    writer = CriticWriter(llm, llm, "Duck Desk", "Quackers", max_repairs=1)
    ep = writer.write(_stories())
    assert llm.stages == ["writer", "critic", "writer-repair-1", "critic"]
    assert ep.segments[-1].text == DEFAULT_OUTRO and ep.story_segments[0].text == GOOD_A
    assert not [f for f in writer.report["fallbacks"] if f["part"] == "outro"]  # the outro may ask to subscribe
    # asked directly to fix the outro, a revision still can't change it
    llm.replies = [dict(NO_CHANGE, outro="A brand new outro.")]
    revised = writer.revise(ep, _stories(), [Issue("outro_problem", "the outro is wrong")], stage="writer-repair-9")
    assert revised.segments[-1].text == DEFAULT_OUTRO


def test_template_intro_changes_by_day_and_never_counts_stories(monkeypatch):
    intros = []
    for offset in range(len(TEMPLATE_HOOKS)):
        _on_day(monkeypatch, date(2026, 9, 29) + timedelta(days=offset))
        intros.append(TemplateWriter("Duck Desk", "Quackers").intro(8).text)
    assert len({_first_words(t) for t in intros}) == len(TEMPLATE_HOOKS)  # a new opening every day of the week
    assert "Quack quack, it's Quackers on Duck Desk!" in " ".join(intros)
    assert not [t for t in intros if re.search(r"\b8\b|\beight\b", t.lower())]


def test_template_intro_skips_openings_a_recent_episode_used(monkeypatch):
    _on_day(monkeypatch, date(2026, 9, 29))
    today = TemplateWriter("Duck Desk", "Quackers").intro(5).text
    tomorrows = TemplateWriter("Duck Desk", "Quackers")
    _on_day(monkeypatch, date(2026, 9, 30))
    tomorrow = tomorrows.intro(5).text
    _on_day(monkeypatch, date(2026, 9, 29))
    fresh = TemplateWriter("Duck Desk", "Quackers", recent_intros=[today, tomorrow]).intro(5).text
    assert _first_words(fresh) not in (_first_words(today), _first_words(tomorrow))
    every_hook = [h.format(host="Quackers", show="Duck Desk") for h in TEMPLATE_HOOKS]
    assert TemplateWriter("Duck Desk", "Quackers", recent_intros=every_hook).intro(5).text  # still an intro


def test_template_intro_skips_a_recent_opening_that_differs_only_in_punctuation(monkeypatch):
    day = date(2026, 9, 29)
    day -= timedelta(days=day.toordinal() % len(TEMPLATE_HOOKS))  # the "Quack quack, it's ..." day
    _on_day(monkeypatch, day)
    assert TemplateWriter("Duck Desk", "Quackers").intro(5).text.startswith("Quack quack, it's Quackers")
    recent = ["Quack quack! It's Quackers, and OpenAI just shipped a pocket robot."]
    intro = TemplateWriter("Duck Desk", "Quackers", recent_intros=recent).intro(5).text
    assert _first_words(intro) != _first_words(recent[0])


def test_template_segment_reads_the_headline_once_and_no_boilerplate():
    story = _story("Nvidia unveils Rubin GPU", summary=(
        "This story originally appeared in The Algorithm, our weekly newsletter on AI. Nvidia unveils Rubin GPU. "
        "The chip runs AI models three times faster than Blackwell, and cloud providers get it first next spring."))
    story.headline = story.title
    seg = template_segment(story)
    assert seg.text.lower().count("nvidia unveils rubin gpu") == 1
    assert seg.text == ("Nvidia unveils Rubin GPU. The chip runs AI models three times faster than Blackwell, and "
                        "cloud providers get it first next spring.")
    story.summary = ("Nvidia has now unveiled its new Rubin GPU line. The chip runs AI models three times faster than "
                     "Blackwell, and cloud providers get it first next spring.")  # a reworded restatement
    seg = template_segment(story)
    assert "unveiled" not in seg.text and seg.text.startswith("Nvidia unveils Rubin GPU. The chip runs")
    ep = TemplateWriter("Duck Desk", "Quackers").write([story])
    assert not [i for i in lint_episode(ep, [story]) if i.code in ("repeats", "source_talk")]
    long_story = _story("OpenAI ships GPT agent for ChatGPT users", summary=(
        "OpenAI ships GPT agent for ChatGPT users in Europe and Asia, starting with paying subscribers today. "
        "It books tables and orders groceries, and it asks before it pays."))  # the headline, then more
    long_story.headline = long_story.title
    seg = template_segment(long_story)
    assert seg.text.lower().count("openai ships gpt agent") == 1 and "It books tables" in seg.text


def test_template_segment_reads_a_short_headline_once_when_the_summary_opens_with_it():
    story = _story("Nvidia unveils Rubin GPU", summary=(
        "Nvidia unveils Rubin GPU at GTC Paris, its fastest chip yet. The chip runs AI models three times faster "
        "than Blackwell, and cloud providers get it first next spring."))
    story.headline = story.title
    assert template_segment(story).text.lower().count("nvidia unveils rubin gpu") == 1


def test_lint_flags_a_short_headline_read_twice_back_to_back():
    story = _story("Nvidia unveils Rubin GPU", summary="Nvidia showed its Rubin GPU at GTC Paris.")
    story.headline = story.title
    ep = TemplateWriter("Duck Desk", "Quackers").write([story])
    ep.segments[1].text = ("Nvidia unveils Rubin GPU. Nvidia unveils Rubin GPU at GTC Paris, its fastest chip yet, "
                           "and cloud providers get the first units next spring for their chatbot services.")
    assert "repeats" in [i.code for i in lint_episode(ep, [story])]


def test_story_block_credits_only_real_outlets():
    story = _story("GPT agent", source="TechCrunch", summary=(
        "To get stories like this in your inbox first, sign up here. OpenAI launched an agent that books trips."))
    story.headline = story.title
    story.outlets = ["TLDR AI", "Hacker News", "The Verge", "r/LocalLLaMA", "Daily Dose", "TechCrunch"]
    only_letters = _story("Chip deal", source="", summary="Nvidia signed a chip deal.")
    only_letters.outlets = ["The Rundown AI", "Superhuman"]
    block = _story_block([story, only_letters], banned={"Daily Dose"})
    assert "Covered by: The Verge, TechCrunch\n" in block
    assert block.count("Covered by") == 1  # a story covered only by newsletters credits nobody
    for name in ("TLDR", "Hacker News", "LocalLLaMA", "Daily Dose", "Rundown", "Superhuman", "inbox", "sign up"):
        assert name not in block, name
    llm = ScriptedLLM(_script(GOOD_A, GOOD_B))
    LLMWriter(llm, "Duck Desk", "Quackers", banned={"Daily Dose"}).draft([story, only_letters])
    assert "Daily Dose" not in llm.calls[0].user and "TLDR" not in llm.calls[0].user


def test_a_critic_quality_finding_sends_the_segment_for_repair():
    flagged = {"intro": [], "outro": [],
               "segments": [{"story": 2, "unsupported": [], "quality": ["only restates the headline"]}]}
    revision = {"intro": "", "outro": "",
                "segments": [{"story": 2, "headline": "Chip is faster", "key_fact": "2x faster", "text": BETTER_B}]}
    llm = ScriptedLLM(_script(GOOD_A, GOOD_B), flagged, revision, NO_ISSUES)
    writer = CriticWriter(llm, llm, "Duck Desk", "Quackers", max_repairs=1)
    ep = writer.write(_stories())
    assert llm.stages == ["writer", "critic", "writer-repair-1", "critic"]
    assert "story 2: only restates the headline" in llm.calls[2].user
    assert [s.text for s in ep.story_segments] == [GOOD_A, BETTER_B]
    assert writer.report["rounds"][0]["critic"] == 1
    # a quality problem the rewrites don't fix falls back to the story's own summary
    llm = ScriptedLLM(_script(GOOD_A, GOOD_B), flagged)
    ep = CriticWriter(llm, llm, "Duck Desk", "Quackers", max_repairs=0).write(_stories())
    assert ep.story_segments[1].text.startswith("The new chip is 2 times faster")


def test_a_story_that_still_reads_source_talk_as_a_template_is_left_out():
    stories = [*_stories(), _lazy_story()]
    forum = ("Over on Reddit, users say GPT-6 now gives shorter answers than GPT-5 did. OpenAI says it is looking "
             "into the reports and will share an update soon. People notice quickly when a model changes.")
    teaser = "Quack! Does GPT-6 feel lazy to you? Users say so, plus a travel agent and a faster chip."
    llm = ScriptedLLM(_script(GOOD_A, GOOD_B, forum, intro=teaser), NO_ISSUES)
    writer = CriticWriter(llm, llm, "Duck Desk", "Quackers", max_repairs=0)
    ep = writer.write(stories)
    assert writer.report["dropped"] == [{"story": 3, "headline": "GPT-6 feels lazy, Reddit users say"}]
    assert ep.stories == stories[:2] and [s.text for s in ep.story_segments] == [GOOD_A, GOOD_B]
    assert "reddit" not in ep.narration.lower() and "GPT-6" not in ep.description
    assert ep.segments[0].text == TemplateWriter("Duck Desk", "Quackers").intro(2).text  # it teased the lost story
    assert {"part": "intro", "why": "teased a story that was left out"} in writer.report["fallbacks"]
    assert ep.segments[-1].text == DEFAULT_OUTRO
    # an intro that didn't tease the dropped story stays
    other = "Quack! A travel agent that books whole trips leads today's show, then a faster chip."
    llm = ScriptedLLM(_script(GOOD_A, GOOD_B, forum, intro=other), NO_ISSUES)
    ep = CriticWriter(llm, llm, "Duck Desk", "Quackers", max_repairs=0).write(stories)
    assert ep.segments[0].text == other and len(ep.story_segments) == 2


def test_a_title_or_description_naming_a_newsletter_or_forum_falls_back():
    script = _script(GOOD_A, GOOD_B)
    script["title"] = "As seen on Hacker News: travel agents and faster chips"
    script["description"] = "Two stories picked by Daily Dose this morning."
    llm = ScriptedLLM(script, NO_ISSUES)
    writer = CriticWriter(llm, llm, "Duck Desk", "Quackers", max_repairs=0, banned={"Daily Dose"})
    ep = writer.write(_stories())
    assert ep.title == TemplateWriter("Duck Desk", "Quackers").write(_stories()).title
    assert ep.description.startswith(default_description(_stories()) + "\n\nSources:\n")
    assert "Daily Dose" not in ep.description and "Hacker News" not in ep.title
    assert {"title", "description"} <= {f["part"] for f in writer.report["fallbacks"]}


def test_write_episode_refuses_a_template_script_unless_allowed():
    def failing():
        return LLMWriter(ScriptedLLM(RuntimeError("model down")), "Duck Desk", "Quackers")

    with pytest.raises(WriterFailed, match="not publishing a template-read script"):
        write_episode(failing(), _stories(), "Duck Desk", "Quackers", allow_template=False)
    custom = f"Subscribe, little ducklings. {SIGN_OFF}"
    ep = write_episode(failing(), _stories(), "Duck Desk", "Quackers", outro=custom)
    assert ep.segments[-1].text == custom and len(ep.story_segments) == 2


# --- pipeline ----------------------------------------------------------------------------------

class Reached(Exception):
    """Raised by a stubbed stage to show the run got that far."""


def _no_fetch(sources_):
    raise AssertionError("fetched news although there is no episode without a model")


def test_live_run_with_no_model_is_refused_before_any_fetch(monkeypatch, tmp_path):
    _clean_env(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "build_sources", lambda cfg: [])
    monkeypatch.setattr(pipeline, "fetch_all", _no_fetch)
    cfg = Config.from_env()
    assert cfg.llm_provider == "none" and cfg.sources == ["newsletter", "rss"]
    with pytest.raises(RuntimeError, match="No AI model is set up"):
        pipeline.run(cfg)


def test_allow_no_ai_or_a_model_lets_the_run_fetch(monkeypatch, tmp_path):
    _clean_env(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "build_sources", lambda cfg: [])

    def fetched(sources_):
        raise Reached("fetching")

    monkeypatch.setattr(pipeline, "fetch_all", fetched)
    monkeypatch.setenv("SHORTS_ALLOW_NO_AI", "on")
    with pytest.raises(Reached):
        pipeline.run(Config.from_env())
    monkeypatch.delenv("SHORTS_ALLOW_NO_AI")
    monkeypatch.setattr(pipeline, "build_llm", lambda cfg, usage: ScriptedLLM())
    with pytest.raises(Reached):
        pipeline.run(Config.from_env())


def _thin(title, **kw):
    return _story(title, summary=f"{title}.", **kw)


def _live_candidates(good_titles):
    letter = _story("Today in AI", url="", kind="newsletter", source="Daily Dose",
                    body="OpenAI shipped an agent. " * 30)
    thin = [_thin("Microsoft Copilot gets AI agents"), _thin("Apple Siri AI upgrade delayed"),
            _story("Show HN: an LLM inference engine in Rust", url="https://github.com/someone/engine", summary="",
                   source="Hacker News")]
    return [letter, *[_news(t, hours_ago=i + 1) for i, t in enumerate(good_titles)], *thin]


def test_run_stops_when_too_few_stories_have_a_real_description(monkeypatch, tmp_path):
    _clean_env(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "build_sources", lambda cfg: [])
    monkeypatch.setattr(pipeline, "fetch_all", lambda s: _live_candidates(GOOD_TITLES[:3]))

    def no_writer(*args, **kwargs):
        raise AssertionError("wrote a script from thin stories")

    monkeypatch.setattr(pipeline, "build_writer", no_writer)
    cfg = replace(Config.from_env(), allow_no_ai=True)
    with pytest.raises(RuntimeError, match=r"Only 3 different stories with a solid description today \(at least 4 needed\)"):
        pipeline.run(cfg)
    assert not (tmp_path / "state" / "seen_urls.json").exists()


def test_run_writes_with_fewer_solid_stories_and_the_writers_inputs(monkeypatch, tmp_path):
    _clean_env(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "build_sources", lambda cfg: [])
    monkeypatch.setattr(pipeline, "fetch_all", lambda s: _live_candidates(GOOD_TITLES[:4]))
    log = IntroLog(tmp_path / "state" / "intros.json")
    log.add("Quack quack, it's Quackers on Duck Desk! Yesterday's hook.")
    seen = {}

    def fake_build_writer(llm, show, host, **kwargs):
        seen["writer"] = kwargs
        return SimpleNamespace(name="fake")

    def fake_write_episode(writer, stories, show, host, **kwargs):
        seen["stories"], seen["write"] = stories, kwargs
        raise Reached("writing")

    monkeypatch.setattr(pipeline, "build_writer", fake_build_writer)
    monkeypatch.setattr(pipeline, "write_episode", fake_write_episode)
    cfg = replace(Config.from_env(), allow_no_ai=True)
    with pytest.raises(Reached):
        pipeline.run(cfg)
    assert sorted(s.title for s in seen["stories"]) == sorted(GOOD_TITLES[:4])  # 4 of 8 is fine; thin ones out
    assert seen["writer"]["recent_intros"] == log.recent
    assert "Daily Dose" in seen["writer"]["banned"] and seen["writer"]["outro"] == DEFAULT_OUTRO
    assert seen["write"]["allow_template"] is True and seen["write"]["outro"] == DEFAULT_OUTRO


def test_intro_log_keeps_the_last_seven_and_survives_a_corrupt_file(tmp_path):
    path = tmp_path / "state" / "intros.json"
    intros = IntroLog(path)
    assert intros.recent == []
    for i in range(9):
        intros.add(f"Hook number {i}")
    assert IntroLog(path).recent == [f"Hook number {i}" for i in range(2, 9)]
    path.write_text('[{"date": "2026-09-28", "text": "Quack')  # a write cut short
    broken = IntroLog(path)
    assert broken.recent == []
    broken.add("Fresh hook")
    assert IntroLog(path).recent == ["Fresh hook"]
    path.write_text(json.dumps(["not a dict", {"date": "2026-09-28"}, {"text": "Kept"}]))
    assert IntroLog(path).recent == ["Kept"]
    path.write_text("")
    assert IntroLog(path).recent == []


def test_intro_log_survives_a_file_that_is_not_a_list(tmp_path):
    path = tmp_path / "intros.json"
    path.write_text("null")
    assert IntroLog(path).recent == []


def test_episode_json_round_trips():
    story = _news("Meta open sources Llama")
    ep = Episode(title="AI today", description=f"Two stories.\n\nSources:\n- Meta open sources Llama: {story.url}",
                 tags=["ai", "llama"],
                 segments=[Segment("intro", "Quack! Llama goes open."),
                           Segment("story", "Meta released Llama with open weights.", "Meta open sources Llama",
                                   "Open weights", "TechCrunch", story.url),
                           Segment("outro", DEFAULT_OUTRO)],
                 stories=[story])
    back = episode_from_json(episode_json(ep))
    assert back == replace(ep, stories=[])  # the stories are left out
    assert episode_json(back) == episode_json(replace(ep, stories=[]))
    data = json.loads(episode_json(ep))
    data["segments"][1]["voice"] = "coral"  # a field this version doesn't know
    assert episode_from_json(json.dumps(data)).segments == ep.segments
