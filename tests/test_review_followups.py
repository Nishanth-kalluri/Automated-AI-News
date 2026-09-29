"""Follow-ups from the review of the video fixes: checks that were too strict for real news, newsletter
links, stories refilled before a day is skipped, and the writer's last-resort paths."""
import json
from dataclasses import replace

import pytest

from shorts import pipeline
from shorts.checks import check_picks, lint_episode, same_event
from shorts.config import DEFAULT_OUTRO
from shorts.content import (clean_text, is_banned_name, is_newsletter_url, publisher_name, repetition,
                            script_problems)
from shorts.models import Segment
from shorts.selection import LLMEditor, SeenStore
from shorts.writer import (CriticWriter, TemplateWriter, WriterFailed, _story_block, build_writer, default_description,
                           description_footer, template_segment, write_episode)
from tests.test_review_fixes import (GOOD_A, GOOD_B, NO_ISSUES, ScriptedLLM, _lazy_story, _news, _script, _stories,
                                     _story)

# --- source text: news that only sounds like boilerplate stays ---------------------------------

@pytest.mark.parametrize("news", [
    "Anthropic updated its consumer terms of use so it can train Claude on user chats unless people opt out.",
    "Meta changed its privacy policy to use conversations with its AI assistant to target ads.",
    "The paper, presented by Google DeepMind researchers at NeurIPS, shows a new training method.",
    "Anthropic's Super Bowl advertisement poked fun at ads in chatbots.",
    "Claude can now read more files at once.",
    "People who subscribe to Google AI Pro get Gemini 3 Deep Think now.",
    "Google's Gemini can now clean up the promotions in your inbox automatically.",
    "Google will train its Gemini models on Reddit posts under a 60 million dollar deal.",
    "Substack launched AI tools that help writers draft their newsletters.",
    "The bill, sponsored by state senator Scott Wiener, passed the California Senate on Monday.",
    "Users can sign up for the waitlist today.",
])
def test_clean_text_keeps_real_news_about_policies_ads_inboxes_and_reddit(news):
    assert clean_text(news) == news


@pytest.mark.parametrize("boilerplate", [
    "Welcome to this week's newsletter.",
    "Today's issue is brought to you by Acme.",
    "Get it delivered to your inbox every morning.",
    "Read full article",
    "Advertisement",
    "The thread was posted on Reddit.",
    "Sign up here.",
])
def test_clean_text_still_drops_the_publication_talking_about_itself(boilerplate):
    news = "OpenAI released a new model for developers on Tuesday."
    assert clean_text(f"{news} {boilerplate}") == news


def test_clean_text_drops_feed_footers_that_are_not_whole_sentences():
    news = "OpenAI released a new model for developers on Tuesday."
    assert clean_text(f"{news} © 2026 TechCrunch. All rights reserved. For personal use only.") == news
    assert clean_text(f"{news} Read full article Comments") == news


# --- the script: news narration is not source talk ---------------------------------------------

@pytest.mark.parametrize("line", [
    "OpenAI will get real-time access to Reddit posts under the deal.",
    "Google pays 60 million dollars a year to train its models on Reddit.",
    "Reddit users will now see AI-written answers in search.",
    "People who subscribe to ChatGPT Plus get the new model first.",
    "Gemini can now sort and summarize the email in your inbox.",
    "AI assistants are creeping into our daily lives.",
    "The US Copyright Office received more than 10,000 comments on AI training.",
    "The FTC points to thousands of public comments on the rule.",
    "Superhuman AI could arrive by 2030, Altman says.",
    "Import AI Chats From ChatGPT Into Claude",
    "Follow-up studies found the same effect.",
])
def test_script_problems_leaves_news_about_reddit_pricing_and_comments_alone(line):
    assert script_problems(line) == []


@pytest.mark.parametrize("line", [
    "People on Reddit loved it.",
    "It got 900 comments on Reddit.",
    "Like and subscribe for daily AI news.",
    "Superhuman AI reports that OpenAI shipped it.",
    "It showed up in your inbox this morning.",
])
def test_script_problems_still_catches_forum_talk_and_calls_to_subscribe(line):
    assert script_problems(line)


def test_a_story_about_reddit_the_company_airs():
    story = _story("Reddit Answers opens to everyone",
                   summary="Reddit made its AI search tool, Reddit Answers, available to all users in the US. It "
                           "writes a short answer from posts and links the threads it used.")
    story.headline = story.title
    text = ("Reddit made its AI search tool, Reddit Answers, free for everyone in the US. You ask a question and it "
            "writes a short answer from posts, with links to its sources. Search is turning into chat everywhere.")
    llm = ScriptedLLM({"title": "Reddit's AI search opens up", "description": "One story.", "tags": [],
                       "intro": "Quack! Reddit's AI search is open to all today.",
                       "segments": [{"headline": "Reddit Answers opens up", "key_fact": "", "text": text}],
                       "outro": ""}, NO_ISSUES)
    writer = CriticWriter(llm, llm, "Duck Desk", "Quackers", max_repairs=0)
    ep = writer.write([story])
    assert writer.report["dropped"] == [] and ep.story_segments[0].text == text


# --- newsletter links and names ----------------------------------------------------------------

def test_newsletter_web_addresses_are_never_credited():
    for name in ("therundown.ai", "theneurondaily.com", "bensbites.com", "tldr.tech", "superhuman.ai"):
        assert is_banned_name(name), name
    for name in ("techcrunch.com", "openai.com", "TechCrunch", "The Verge", "MIT Technology Review"):
        assert not is_banned_name(name), name
    for url in ("https://www.bensbites.com/p/x", "https://importai.substack.com/p/x", "https://theneuron.ai/x"):
        assert is_newsletter_url(url) and publisher_name(url) == "", url
    assert not is_newsletter_url("https://techcrunch.com/2026/09/29/x/")


def test_a_pick_linking_to_a_newsletter_page_is_not_news(tmp_path):
    letter = _story("Rundown", url="", kind="newsletter", source="The Rundown AI",
                    body="OpenAI ships agent https://www.therundown.ai/p/openai-agent " * 5)
    pick = replace(_news("OpenAI ships GPT agent"), url="https://www.therundown.ai/p/openai-agent")
    pick.headline = pick.title
    issues = check_picks([pick], [letter], 1, set(), [], 48)
    assert [i.code for i in issues if i.fatal] == ["not_news"]


def test_the_editor_drops_a_newsletter_link_and_keeps_the_story(tmp_path):
    reply = {"stories": [{"headline": "OpenAI ships GPT agent", "summary": "OpenAI launched an agent.", "key_fact": "",
                          "url": "https://www.therundown.ai/p/openai-agent", "outlets": ["The Rundown"], "why": ""}]}
    stories = LLMEditor.to_stories(reply, [], 1, quiet=True)
    assert stories[0].url == "" and stories[0].source == ""


def test_too_few_picks_says_the_real_minimum():
    picks = [replace(_news("OpenAI ships GPT agent"), headline="OpenAI ships GPT agent")]
    issue = next(i for i in check_picks(picks, picks, 8, set(), [], 48, min_n=4) if i.code == "too_few")
    assert "at least 4" in issue.detail and "up to 8" in issue.detail


def test_the_sources_list_never_names_a_newsletter_or_links_a_forum():
    a, b = _stories()
    a.url = "https://news.ycombinator.com/item?id=1"
    b.headline = "The Download: Chip is faster"
    footer = description_footer([a, b])
    assert "ycombinator" not in footer and "The Download" not in footer
    assert "- Lab ships agent\n" in footer and f"- {b.url}\n" in footer


# --- same event, repetition --------------------------------------------------------------------

@pytest.mark.parametrize("a, b", [
    ("Meta's Ray-Ban Display glasses fail in live demo", "Meta's Ray-Ban Display glasses go on sale"),
    ("OpenAI's ChatGPT Atlas browser launches on Mac", "OpenAI's ChatGPT Atlas browser has a prompt injection flaw"),
])
def test_two_things_happening_to_one_product_are_different_events(a, b):
    assert not same_event(a, b)


def test_the_run_two_duplicate_is_still_one_event():
    assert same_event("Nvidia launches new platform for reining in rogue AI agents",
                      "Nvidia says its new AI safety platform can contain rogue agents within ‘milliseconds’")


def test_parallel_facts_with_different_numbers_are_not_a_repeat():
    assert repetition("Sonnet now costs 3 dollars per million input tokens. "
                      "Opus now costs 15 dollars per million input tokens.") == ""
    assert repetition("Sonnet now costs 3 dollars per million tokens. Sonnet now costs 3 dollars per million tokens.")


def test_saying_the_sign_off_inside_a_story_is_a_repeat():
    a, b = _stories()
    ep = TemplateWriter("Duck Desk", "Quackers").write([a, b])
    segs = list(ep.segments)
    segs[2] = replace(segs[2], text=GOOD_B + " That's the news from the pond.")
    issues = lint_episode(replace(ep, segments=segs), [a, b])
    assert any(i.code == "repeats" and i.index == 1 and "sign-off" in i.detail for i in issues)


# --- the writer's last resorts -----------------------------------------------------------------

def test_the_template_reads_a_headline_restated_in_other_words_once():
    story = _story("OpenAI launches GPT-6",
                   summary="OpenAI has launched GPT-6, its newest and largest model. It is available to paying "
                           "ChatGPT users today and to developers next week.")
    story.headline = story.title
    text = template_segment(story).text
    assert text.startswith("OpenAI has launched GPT-6") and text.count("GPT-6") == 1


def test_a_template_fallback_that_copies_the_article_is_left_out():
    a, b = _stories()
    b.body = b.summary + " More text from the article follows here."
    no_change = {"intro": "", "outro": "", "segments": []}
    flagged = {"intro": [], "outro": [], "segments": [{"story": 2, "unsupported": ["5 times"], "quality": []}]}
    llm = ScriptedLLM(_script(GOOD_A, GOOD_B.replace("2 times", "5 times")), flagged, no_change, flagged)
    writer = CriticWriter(llm, llm, "Duck Desk", "Quackers", max_repairs=1)
    ep = writer.write([a, b])
    assert [d["story"] for d in writer.report["dropped"]] == [2] and len(ep.story_segments) == 1


def test_a_dropped_story_leaves_no_trace_in_the_intro_title_or_description():
    stories = [*_stories(), _lazy_story()]
    forum = ("Over on Reddit, users say GPT-6 now gives shorter answers than GPT-5 did. OpenAI says it is looking "
             "into the reports and will share an update soon. People notice quickly when a model changes.")
    script = _script(GOOD_A, GOOD_B, forum, intro="Quack! Is GPT-6 lazy? Plus a travel agent and a faster chip.")
    script["title"] = "Lazy GPT-6, travel agents and faster chips"
    script["description"] = "Is GPT-6 lazy? Also a travel agent and a faster chip."
    llm = ScriptedLLM(script, NO_ISSUES)
    writer = CriticWriter(llm, llm, "Duck Desk", "Quackers", max_repairs=0)
    ep = writer.write(stories)
    template = TemplateWriter("Duck Desk", "Quackers")
    assert ep.segments[0].text == template.intro(2).text
    assert ep.title == template.write(stories[:2]).title and "GPT-6" not in ep.title
    assert ep.description.startswith(default_description(stories[:2]) + "\n\nSources:\n")
    assert "GPT-6" not in ep.description


def test_title_and_description_problems_cost_no_rewrite_rounds():
    script = _script(GOOD_A, GOOD_B)
    script["title"] = "TLDR AI: travel agents and faster chips"
    script["description"] = "Two stories today. Subscribe for daily AI news!"
    llm = ScriptedLLM(script, NO_ISSUES)
    writer = CriticWriter(llm, llm, "Duck Desk", "Quackers", max_repairs=2)
    ep = writer.write(_stories())
    assert llm.stages == ["writer", "critic"]
    assert "TLDR" not in ep.title and "Subscribe" not in ep.description
    whys = {f["part"]: f["why"] for f in writer.report["fallbacks"]}
    assert whys["title"] == "names TLDR AI" and whys["description"] == "asks people to subscribe"


def test_without_agents_the_script_still_gets_the_code_checks():
    forum = ("People on Hacker News loved the new chip, which runs AI models 2 times faster at inference. Serving "
             "chatbots gets cheaper, and the first servers ship to customers this year.")
    llm = ScriptedLLM(_script(GOOD_A, forum))
    writer = build_writer(llm, "Duck Desk", "Quackers", agents=False)
    ep = write_episode(writer, _stories(), "Duck Desk", "Quackers", allow_template=True)
    assert writer.name == "llm" and llm.stages == ["writer"]  # one call: no critic, no rewrites
    assert "Hacker News" not in ep.narration
    assert ep.story_segments[0].text == GOOD_A


def test_a_script_that_is_mostly_template_text_is_not_published():
    draft = _script(GOOD_A, GOOD_B)
    draft["segments"] = []  # the writer left every story out
    llm = ScriptedLLM(draft, NO_ISSUES)
    writer = CriticWriter(llm, llm, "Duck Desk", "Quackers", max_repairs=0)
    with pytest.raises(WriterFailed, match="only 0 of 2 stories"):
        write_episode(writer, _stories(), "Duck Desk", "Quackers", allow_template=False)
    llm = ScriptedLLM(draft, NO_ISSUES)
    writer = CriticWriter(llm, llm, "Duck Desk", "Quackers", max_repairs=0)
    assert len(write_episode(writer, _stories(), "Duck Desk", "Quackers").story_segments) == 2  # allowed offline


def test_the_template_fallback_still_avoids_recent_intros():
    recent = [TemplateWriter("Duck Desk", "Quackers").intro(2).text]
    llm = ScriptedLLM(RuntimeError("model down"))
    writer = build_writer(llm, "Duck Desk", "Quackers", agents=True, recent_intros=recent)
    ep = write_episode(writer, _stories(), "Duck Desk", "Quackers", allow_template=True)
    assert ep.segments[0].text != recent[0]


def test_the_writer_sees_cleaned_article_text():
    a, b = _stories()
    a.body = ("This story originally appeared in The Algorithm, our weekly newsletter on AI. The lab shipped an "
              "agent that books travel in 3 steps.")
    block = _story_block([a, b])
    assert "originally appeared" not in block and "The lab shipped an agent" in block


# --- the run: refill before skipping, public state ---------------------------------------------

class _Reader:
    name = "fake"

    def __init__(self):
        self.read = []

    def enrich(self, stories):
        self.read += [s.title for s in stories]


def test_a_list_shortened_by_the_checks_is_refilled_before_the_day_is_skipped(tmp_path):
    titles = ["OpenAI ships GPT agent", "EU passes AI audit rules", "Google Gemini tops math olympiad",
              "Meta open sources Llama", "Anthropic raises funding for Claude"]
    candidates = [_news(t, hours_ago=i + 1) for i, t in enumerate(titles)]
    kept = [replace(c, headline=c.title) for c in candidates[:3]]
    reader = _Reader()
    got = pipeline._refill(kept, 4, 8, candidates, SeenStore(tmp_path / "seen.json"), 48, reader, None)
    assert len(got) >= 4 and [s.title for s in got[:3]] == titles[:3]
    assert reader.read and set(reader.read) <= set(titles[3:])  # only the new stories were read


def test_refill_with_nothing_left_keeps_the_list(tmp_path):
    candidates = [_news(t) for t in ["OpenAI ships GPT agent", "EU passes AI audit rules"]]
    kept = [replace(c, headline=c.title) for c in candidates]
    reader = _Reader()
    assert pipeline._refill(kept, 4, 8, candidates, SeenStore(tmp_path / "seen.json"), 48, reader, None) == kept
    assert reader.read == []


def test_outro_card_asks_to_subscribe():
    from shorts import visuals
    source = open(visuals.__file__).read()
    assert "Subscribe for tomorrow's" in source and "Follow for tomorrow's" not in source
    assert "Subscribe" in DEFAULT_OUTRO


def test_last_episode_state_holds_the_script_not_the_articles(tmp_path):
    from shorts.writer import episode_from_json, episode_json
    ep = TemplateWriter("Duck Desk", "Quackers").write(_stories())
    ep.stories[0].body = "ARTICLE TEXT"
    text = episode_json(replace(ep, stories=[]))
    assert "ARTICLE TEXT" not in text and json.loads(text)["stories"] == []
    assert [s.text for s in episode_from_json(text).segments] == [s.text for s in ep.segments]
    assert isinstance(ep.segments[1], Segment)
