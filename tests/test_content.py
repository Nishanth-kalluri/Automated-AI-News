from dataclasses import replace

import pytest

from shorts.checks import check_picks, lint_episode, same_event
from shorts.config import DEFAULT_OUTRO
from shorts.content import (MIN_DESCRIPTION_WORDS, MIN_NEW_WORDS, banned_names, clean_text, copied_run,
                            description_problem, is_aggregator_url, is_banned_name, publisher_name, repetition,
                            says_little, script_problems)
from shorts.models import Episode, Evidence, Segment, Story
from shorts.writer import TemplateWriter
from tests.test_agents import NOW, _story

# The footer MIT Technology Review's feed puts under its stories; video 2 read it out loud.
ALGORITHM_FOOTER = ("This story originally appeared in The Algorithm, our weekly newsletter on AI. To get stories "
                    "like this in your inbox first, sign up here.")
NEWS = "OpenAI is testing ads in ChatGPT for free users in the US, starting with shopping searches."

NV_HEADLINE = "Nvidia launches new platform for reining in rogue AI agents"
NV_SAYS = "Nvidia says its new AI safety platform can contain rogue agents within ‘milliseconds’"
OPENAI_ROGUE = "OpenAI still doesn't seem to have a handle on all of its rogue AI activity"
NV_SUMMARY = ("Nvidia built a safety layer that watches AI agents at work and can stop one within milliseconds if it "
              "misbehaves.")
NV_BODY = ("Nvidia on Monday introduced a software platform that monitors autonomous AI agents inside corporate "
           "networks and can halt an agent that starts acting outside its permissions within milliseconds, the "
           "company said. Early customers include banks and hospital groups that run hundreds of agents at once.")
AD_TITLE = "OpenAI is testing ads in ChatGPT"
AD_SUMMARY = ("OpenAI will start showing ads to free ChatGPT users in the US, beginning with shopping searches, while "
              "paid plans stay ad free.")
AD_BODY = ("OpenAI will begin showing advertisements to people who use the free version of ChatGPT in the United "
           "States, next to shopping searches. Paid subscribers will not see ads, and the company says advertisers "
           "never get access to conversations.")


# --- clean_text ------------------------------------------------------------------------------

def test_clean_text_drops_the_newsletter_footer_and_keeps_the_news():
    assert clean_text(f"{NEWS} {ALGORITHM_FOOTER}") == NEWS
    assert clean_text(ALGORITHM_FOOTER) == ""
    assert clean_text("") == ""


@pytest.mark.parametrize("boilerplate", [
    "The post OpenAI is testing ads in ChatGPT appeared first on AI News.",
    "412 points, 88 comments on HN",
    "Discussed on Hacker News.",
    "The thread has 2,000 upvotes.",
    "Subscribe to our podcast for more.",
    "Sign up for our free daily briefing.",
    "Read more at TechCrunch.",
    "This article was first published on Medium.",
    "Sponsored by Acme Cloud.",
])
def test_clean_text_drops_sentences_about_the_publication(boilerplate):
    assert clean_text(f"{NEWS} {boilerplate}") == NEWS


@pytest.mark.parametrize("mark", ["[…]", "[...]", "[&#8230;]"])
def test_clean_text_drops_feed_ellipsis_markers(mark):
    assert clean_text(f"Anthropic brought Claude for Chrome to more users {mark}") == \
        "Anthropic brought Claude for Chrome to more users"


def test_clean_text_keeps_news_that_only_sounds_like_boilerplate():
    news = ("ChatGPT now has 20 million paying subscribers. Developers can sign up for the beta starting Monday. "
            "The new model improved 12 points on the SWE-bench coding benchmark.")
    assert clean_text(news) == news


def test_clean_text_keeps_news_about_state_sponsored_hackers_and_long_context():
    for news in ("OpenAI says state-sponsored hackers from China and Iran used ChatGPT to write malware.",
                 "Gemini can now read more than a million tokens of code at once."):
        assert clean_text(news) == news


# --- publisher names -------------------------------------------------------------------------

@pytest.mark.parametrize("feed_title, name", [
    ("AI News & Artificial Intelligence | TechCrunch", "TechCrunch"),
    ("AI | The Verge", "The Verge"),
    ("Artificial intelligence – MIT Technology Review", "MIT Technology Review"),
])
def test_publisher_name_takes_the_outlet_from_the_feed_title(feed_title, name):
    assert publisher_name("https://feeds.example.org/story/1", feed_title) == name


def test_publisher_name_knows_outlets_by_domain_and_subdomain():
    assert publisher_name("https://techcrunch.com/2026/09/29/nvidia-agents/") == "TechCrunch"
    assert publisher_name("https://www.theverge.com/ai/openai-ads") == "The Verge"
    assert publisher_name("https://edition.cnn.com/tech/ai") == "CNN"
    assert publisher_name("https://www.technologyreview.com/2026/x/", "AI | Some Blog") == "MIT Technology Review"
    assert publisher_name("https://notwired.com/story") == "notwired.com"  # a look-alike domain is not Wired


def test_publisher_name_falls_back_to_the_host():
    assert publisher_name("https://www.example-ai-blog.org/post/1") == "example-ai-blog.org"
    assert publisher_name("https://example.org/post/1", "Hacker News: Front Page") == "example.org"
    assert publisher_name("https://example.org/post/1", "Top | Hacker News") == "example.org"
    assert publisher_name("", "") == ""


@pytest.mark.parametrize("url", [
    "https://news.ycombinator.com/item?id=41234567",
    "https://www.reddit.com/r/OpenAI/comments/abc/rogue_ai/",
    "https://old.reddit.com/r/LocalLLaMA/",
    "https://www.therundown.ai/p/openai-ads",
    "https://tldr.tech/ai/2026-09-29",
    "https://ai-weekly.beehiiv.com/p/issue-12",
])
def test_publisher_name_never_credits_forums_or_newsletters(url):
    assert publisher_name(url) == ""
    assert publisher_name(url, "AI | The Verge") == ""


# --- banned names and forum links ------------------------------------------------------------

def test_is_banned_name_covers_newsletters_forums_and_todays_senders():
    for name in ("The Rundown AI", "TLDR AI", "Superhuman", "Ben's Bites", "The Algorithm", "Hacker News", "Reddit",
                 "r/OpenAI", "r/LocalLLaMA"):
        assert is_banned_name(name), name
    for name in ("TechCrunch", "The Verge", "MIT Technology Review", "Reuters", "", "   "):
        assert not is_banned_name(name), name
    assert not is_banned_name("Daily AI Digest")
    assert is_banned_name("Daily AI Digest", ("Daily AI Digest",))


def test_is_aggregator_url_matches_discussion_threads_only():
    for url in ("https://news.ycombinator.com/item?id=41234567", "https://www.reddit.com/r/OpenAI/comments/abc/x/",
                "https://old.reddit.com/r/LocalLLaMA/", "https://redd.it/abc", "https://www.techmeme.com/260929/p1"):
        assert is_aggregator_url(url), url
    for url in ("https://techcrunch.com/2026/09/29/x/", "https://notreddit.com/x", "", "not a link", "https://[bad"):
        assert not is_aggregator_url(url), url


def test_banned_names_are_todays_newsletter_senders():
    candidates = [_story("Issue 1", kind="newsletter", source="Daily AI Digest"),
                  _story("Issue 2", kind="newsletter", source=" The Prompt "),
                  _story("Issue 3", kind="newsletter", source="  "),
                  _story("Nvidia agents", source="TechCrunch")]
    assert banned_names(candidates) == {"Daily AI Digest", "The Prompt"}


# --- what the host may say -------------------------------------------------------------------

@pytest.mark.parametrize("line, why", [
    ("This came from a newsletter I read this morning.", "mentions a newsletter"),
    ("This story originally appeared in MIT Technology Review.", "says where the story originally appeared"),
    ("The post appeared first on AI News.", "says where the story first appeared"),
    ("The thread hit 412 points and 88 comments.", "reads out forum points or comments"),
    ("It already has 1.2k upvotes.", "reads out forum upvotes"),
    ("People on Hacker News loved it.", "names Hacker News"),
    ("It was the top post on HN today.", "names Hacker News"),
    ("Reddit users are split on it.", "names Reddit"),
    ("Over on r/LocalLLaMA, people ran it at home.", "names Reddit"),
    ("In our weekly roundup, OpenAI leads.", "speaks as a publication"),
    ("As we reported yesterday, the deal closed.", "speaks as a publication"),
    ("It showed up in your inbox this morning.", "talks about an inbox"),
    ("The comment section went wild.", "talks about a comment thread"),
    ("The Rundown says OpenAI shipped it.", "names The Rundown"),
    ("TLDR: OpenAI shipped it.", "names TLDR"),
    ("The Algorithm had the story first.", "names The Algorithm"),
])
def test_script_problems_flags_source_talk(line, why):
    assert why in script_problems(line)
    assert why in script_problems(line, story=False)


def test_script_problems_names_todays_newsletters():
    assert script_problems("Daily AI Digest says OpenAI shipped it.", {"Daily AI Digest"}) == ["names Daily AI Digest"]
    assert script_problems("Daily AI Digest says OpenAI shipped it.") == []
    assert script_problems("AI moves fast.", {"AI"}) == []  # names this short can't be matched safely


def test_only_the_outro_may_ask_people_to_subscribe():
    for line in ("Subscribe for more AI news!", "Follow for more.", "Follow us for daily updates."):
        assert script_problems(line) == ["asks people to subscribe"], line
        assert script_problems(line, story=False) == [], line
    assert "subscribe" in DEFAULT_OUTRO.lower() and script_problems(DEFAULT_OUTRO, story=False) == []


def test_script_problems_leaves_normal_news_alone():
    for line in ("The new model improved 12 points on the SWE-bench coding benchmark.",
                 "ChatGPT now has 20 million paying subscribers.",
                 "Critics commented that the deal gives Nvidia too much power.",
                 "You can sign up for the waitlist today.",
                 "Researchers reported the result in a paper on Monday.",
                 "The subscription still costs 20 dollars a month.",
                 "It follows a similar move by Google last week.",
                 "Quack quack, it's Quackers on Duck Desk!"):
        assert script_problems(line) == [], line


def test_script_problems_does_not_mistake_everyday_phrases_for_newsletter_names():
    for line in ("China can no longer import AI chips from Nvidia without a license.",
                 "DeepMind says the model reached superhuman performance at chess.",
                 "Meta changed the algorithm behind Instagram's feed."):
        assert script_problems(line) == [], line


# --- repetition, thin segments and copying ---------------------------------------------------

def test_repetition_catches_the_title_description_title_segment_from_the_video():
    detail = "Free users in the US will see ads first, next to shopping searches, while paid plans stay ad free."
    assert repetition(f"{AD_TITLE}. {detail} {AD_TITLE}.", AD_TITLE) == "says the same sentence twice"
    assert repetition(f"{AD_TITLE}. {detail}", AD_TITLE) == ""


AD_HEADLINE = "OpenAI tests ads in ChatGPT"
# The old template: the headline, then a feed summary that opens with the same headline.
HEADLINE_TWICE = (f"{AD_HEADLINE}. {AD_HEADLINE} for free users in the US, starting with shopping searches, while "
                  "paid plans stay ad free and advertisers never see chats.")


def test_repetition_catches_the_headline_read_twice():
    text = ("OpenAI tests ads in ChatGPT, and free users in the US see them first. Paid plans stay ad free, so for "
            "now OpenAI tests ads in ChatGPT only on the free tier.")
    assert repetition(text, AD_HEADLINE) == "reads the headline twice"
    assert repetition(f"{AD_TITLE}. {AD_TITLE} for free users in the US.", AD_TITLE)  # 6 words: caught as a run


def test_repetition_catches_a_short_headline_read_twice_in_a_row():
    assert repetition(HEADLINE_TWICE, AD_HEADLINE) == "reads the headline twice"
    assert ("repeats", 1) in _fatal(_episode(ad=HEADLINE_TWICE))


def test_repetition_catches_a_sentence_or_a_six_word_run_said_twice():
    assert repetition("Nvidia built a safety layer for AI agents. It watches what they do. "
                      "Nvidia built a safety layer for AI agents.") == "says the same sentence twice"
    assert repetition("The chip runs AI models twice as fast. Cloud providers say it runs AI models twice as fast "
                      "for less money.") == 'repeats "runs ai models twice as fast"'


def test_says_little_needs_ten_words_beyond_the_headline():
    extra = "banks hospitals and insurers test software that halts misbehaving bots quickly".split()
    assert says_little(f"{NV_HEADLINE}. Big news today.", NV_HEADLINE)
    assert says_little(f"{NV_HEADLINE}. {' '.join(extra[:MIN_NEW_WORDS - 1])}.", NV_HEADLINE)
    assert not says_little(f"{NV_HEADLINE}. {' '.join(extra[:MIN_NEW_WORDS])}.", NV_HEADLINE)
    assert not says_little(NV_SUMMARY, NV_HEADLINE)


def test_copied_run_flags_15_words_verbatim_but_not_14():
    words = NV_BODY.split()
    copied = copied_run(f"Here's the news: {' '.join(words[:15])}, and more.", [AD_BODY, NV_BODY])
    assert copied == " ".join(words[:15]).lower()
    assert copied_run(f"Here's the news: {' '.join(words[:14])}, and more.", [NV_BODY]) == ""
    assert copied_run(f"HERE'S THE NEWS: {' '.join(words[:15]).upper()}!", [NV_BODY])  # case and commas don't hide it
    assert copied_run(" ".join(words[:20]), ["", AD_BODY]) == ""


# --- which stories may air -------------------------------------------------------------------

def _nv(**kw):
    fields = dict(title=NV_HEADLINE, url="https://techcrunch.com/2026/09/29/nvidia-agents/", source="TechCrunch",
                  published=NOW, summary=NV_SUMMARY)
    return Story(**{**fields, **kw})


def test_description_problem_passes_a_story_with_a_real_description():
    assert description_problem(_nv()) == ""
    assert description_problem(_nv(title="x", headline="Nvidia reins in rogue AI agents")) == ""


@pytest.mark.parametrize("summary", [
    "",
    f"{NV_HEADLINE}.",
    f"{NV_HEADLINE} today, the company said on Monday.",
    "412 points, 88 comments on HN",
    f"Nvidia built a safety layer for agents. {ALGORITHM_FOOTER}",
])
def test_description_problem_rejects_a_story_with_only_a_headline(summary):
    assert description_problem(_nv(summary=summary)) == "has no real description, only a headline"


def test_description_problem_needs_enough_words_and_enough_beyond_the_headline():
    new = "chip maker software halts misbehaving bots within milliseconds at banks hospitals and insurers".split()
    assert description_problem(_nv(summary=" ".join(new[:MIN_DESCRIPTION_WORDS]))) == ""
    assert description_problem(_nv(summary=" ".join(new[:MIN_DESCRIPTION_WORDS - 1])))
    assert description_problem(_nv(summary=f"{NV_HEADLINE} {' '.join(new[:MIN_NEW_WORDS])}")) == ""
    assert description_problem(_nv(summary=f"{NV_HEADLINE} {' '.join(new[:MIN_NEW_WORDS - 1])}"))


def test_checked_evidence_or_the_article_rescues_a_thin_summary():
    thin = _nv(summary=NV_HEADLINE)
    quote = Evidence("The platform can halt an agent within milliseconds.", thin.url)
    assert description_problem(replace(thin, evidence=[quote])) == ""
    assert description_problem(replace(thin, body=" ".join(["word"] * 80))) == ""
    assert description_problem(replace(thin, body=" ".join(["word"] * 79)))
    assert description_problem(replace(thin, body=f"{ALGORITHM_FOOTER} " * 5))  # boilerplate isn't an article


def test_description_problem_never_airs_a_forum_thread_or_a_story_without_a_headline():
    for url in ("https://news.ycombinator.com/item?id=41234567", "https://www.reddit.com/r/OpenAI/comments/abc/x/"):
        story = _nv(url=url, evidence=[Evidence("quote", url)], body=NV_BODY * 3)
        assert description_problem(story) == "links to a discussion thread, not a news story"
    assert description_problem(_nv(title="", headline="")) == "has no headline"


# --- duplicates and picks --------------------------------------------------------------------

def test_same_event_matches_the_two_nvidia_rogue_agent_stories():
    assert same_event(NV_HEADLINE, NV_SAYS) and same_event(NV_SAYS, NV_HEADLINE)
    assert not same_event(NV_HEADLINE, OPENAI_ROGUE) and not same_event(NV_SAYS, OPENAI_ROGUE)
    assert not same_event("Nvidia launches new platform for training robots in simulation", NV_SAYS)


def test_same_event_keeps_the_same_launch_by_different_companies_apart():
    assert not same_event("Microsoft launches new platform for reining in rogue AI agents", NV_HEADLINE)


def test_check_picks_drops_the_second_nvidia_story_today_and_after_it_aired():
    picks = [_story(NV_HEADLINE), _story(NV_SAYS), _story(OPENAI_ROGUE)]
    issues = check_picks(picks, picks, 3, set(), [], max_age_hours=30)
    assert [(i.code, i.index) for i in issues if i.fatal] == [("duplicate", 1)]
    later = [_story(NV_SAYS)]
    issues = check_picks(later, later, 1, set(), [NV_HEADLINE], max_age_hours=30)
    assert [(i.code, i.index) for i in issues if i.fatal] == [("already_aired", 0)]


def test_check_picks_rejects_links_to_forum_threads():
    hn = _story("Show HN: a sandbox for local agents", url="https://news.ycombinator.com/item?id=41234567",
                source="Hacker News")
    news = _story(NV_HEADLINE, url="https://techcrunch.com/2026/09/29/nvidia-agents/")
    thread = _story("A thread about GPU prices", url="https://www.reddit.com/r/LocalLLaMA/comments/abc/gpus/")
    issues = check_picks([hn, news, thread], [hn, news, thread], 3, set(), [], max_age_hours=30)
    assert [(i.index, i.fatal) for i in issues if i.code == "not_news"] == [(0, True), (2, True)]


def test_check_picks_asks_for_more_stories_only_below_min_n():
    picks = [_story(t) for t in ("OpenAI ships GPT agent", "Nvidia unveils inference chip", "EU passes AI audit rules",
                                 "Meta open sources Llama")]

    def too_few(k, **kw):
        return [i for i in check_picks(picks[:k], picks, 6, set(), [], 30, **kw) if i.code == "too_few"]

    assert too_few(4, min_n=4) == []
    short = too_few(3, min_n=4)
    assert len(short) == 1 and short[0].fatal and short[0].index is None
    assert too_few(4)  # without min_n, n is the minimum


# --- the script lint -------------------------------------------------------------------------

NV_TEXT = ("Nvidia has a new safety system for AI agents. It watches what each agent does at work, and if one starts "
           "misbehaving, it can shut it down within milliseconds. Banks and hospitals are among the first to try it.")
AD_TEXT = ("ChatGPT is getting ads. Free users in the US will see them first, mostly beside shopping searches, while "
           "people on paid plans stay ad free. OpenAI says advertisers never see your chats.")
INTRO = "Quack quack, it's Quackers on Duck Desk! The AI pond is busy today."
FRAME = "Duck Desk Quackers Tuesday, September 29, 2026"
DESCRIPTION = "Nvidia's safety system for AI agents, and ads in ChatGPT."


def _stories():
    return [_nv(body=NV_BODY, headline="Nvidia reins in rogue AI agents"),
            Story(AD_TITLE, "https://www.theverge.com/ai/openai-ads", "The Verge", NOW, summary=AD_SUMMARY,
                  body=AD_BODY, headline=AD_HEADLINE)]


def _episode(nv=NV_TEXT, ad=AD_TEXT, intro=INTRO, outro=DEFAULT_OUTRO, title="Nvidia reins in rogue agents, and ads"):
    return Episode(title, DESCRIPTION, ["ai"], [
        Segment("intro", intro),
        Segment("story", nv, "Nvidia reins in rogue AI agents", "Stops agents in milliseconds", "TechCrunch"),
        Segment("story", ad, "OpenAI tests ads in ChatGPT", "Free users see ads first", "The Verge"),
        Segment("outro", outro)])


def _issues(episode, stories=None, **kw):
    return lint_episode(episode, stories or _stories(), FRAME, **kw)


def _fatal(episode, stories=None, **kw):
    return {(i.code, i.index) for i in _issues(episode, stories, **kw) if i.fatal}


def test_a_clean_episode_passes_the_lint():
    assert _issues(_episode(), recent_intros=["Splash! Quackers here with today's Duck Desk."],
                   description=DESCRIPTION) == []


@pytest.mark.parametrize("last_line", [
    "This story originally appeared in The Algorithm, our weekly newsletter.",
    "It was the top story on Hacker News today.",
    "Redditors on r/LocalLLaMA are already testing it.",
    "Subscribe for more stories like this.",
])
def test_lint_flags_source_talk_in_a_story(last_line):
    ep = _episode(nv=NV_TEXT.replace("Banks and hospitals are among the first to try it.", last_line))
    assert ("source_talk", 0) in _fatal(ep)


def test_lint_flags_source_talk_in_the_headline_key_fact_and_todays_newsletter_names():
    ep = _episode()
    ep.segments[1].headline = "Nvidia agents top Hacker News"
    assert ("source_talk", 0) in _fatal(ep)
    ep = _episode()
    ep.segments[2].key_fact = "412 points, 88 comments"
    assert ("source_talk", 1) in _fatal(ep)
    ep = _episode(ad=AD_TEXT.replace("OpenAI says", "Daily AI Digest says"))
    assert ("source_talk", 1) not in _fatal(ep)
    assert ("source_talk", 1) in _fatal(ep, banned={"Daily AI Digest"})


def test_lint_flags_a_segment_that_repeats_itself():
    title_twice = f"{AD_TITLE}. Free users in the US will see ads first, next to shopping searches. {AD_TITLE}."
    assert ("repeats", 1) in _fatal(_episode(ad=title_twice))
    run_twice = ("ChatGPT is getting ads for free users. Starting today, ChatGPT is getting ads for free users in the "
                 "US, mostly beside shopping searches, while paid plans stay ad free.")
    assert ("repeats", 1) in _fatal(_episode(ad=run_twice))


def test_lint_flags_a_segment_that_barely_goes_beyond_its_headline():
    assert ("says_little", 1) in _fatal(_episode(ad="OpenAI tests ads in ChatGPT. Big news for ChatGPT today."))


def test_lint_flags_narration_copied_from_the_article_or_its_evidence():
    first_sentence = NV_BODY.split(". ")[0] + "."
    assert ("copied", 0) in _fatal(_episode(nv=first_sentence))
    stories = _stories()
    quote = "Paid subscribers will not see ads, and the company says advertisers never get access to conversations"
    stories[1] = replace(stories[1], body="", evidence=[Evidence(quote, stories[1].url)])
    ad = f"ChatGPT is getting ads for free users in the US. {quote}. It is a big change for the chatbot."
    assert ("copied", 1) in _fatal(_episode(ad=ad), stories)
    assert ("copied", 1) not in _fatal(_episode(), stories)


def test_lint_flags_an_intro_that_opens_like_a_recent_one():
    recent = ["Quack quack! It's Quackers on Duck Desk, and the pond is buzzing."]
    problems = [i for i in _issues(_episode(), recent_intros=recent) if i.code == "intro_problem"]
    assert len(problems) == 1 and problems[0].fatal and problems[0].index is None
    assert "opens with" in problems[0].detail and "fresh hook" in problems[0].detail
    assert not [i for i in _issues(_episode(), recent_intros=["Splash! Quackers here."]) if i.code == "intro_problem"]


def test_lint_checks_the_intro_and_outro_for_source_talk():
    assert ("intro_problem", None) in _fatal(_episode(intro="Welcome to Duck Desk, our weekly newsletter on AI!"))
    assert ("outro_problem", None) in _fatal(_episode(outro="Thanks to Hacker News for the tips. See you tomorrow!"))
    assert ("outro_problem", None) not in _fatal(_episode(outro=DEFAULT_OUTRO))


def test_the_intro_may_not_ask_people_to_subscribe():
    assert ("intro_problem", None) in _fatal(_episode(intro="Quack! Hit subscribe, then let's dive into AI today."))


def test_lint_flags_a_title_or_description_that_names_a_newsletter_or_forum():
    issues = _issues(_episode(title="Today's top AI stories from TLDR"),
                     description="The best of Hacker News and The Rundown newsletter.")
    found = {i.code: i for i in issues if i.code in ("title_problem", "description_problem")}
    assert set(found) == {"title_problem", "description_problem"}
    assert all(i.fatal and i.index is None for i in found.values())
    assert "names TLDR" in found["title_problem"].detail
    assert not [i for i in _issues(_episode(), description=None) if i.code == "description_problem"]
    ep = _episode()
    ep.description = "Stories from The Rundown newsletter."  # only the description passed in is checked
    assert not [i for i in _issues(ep) if i.code == "description_problem"]


def test_the_template_intro_never_opens_like_a_recent_intro_the_lint_would_reject():
    stories = _stories()[:1]
    today = TemplateWriter("Duck Desk", "Quackers").intro(1).text
    recent = [today.replace(",", "").replace("!", ".")]  # same words, different punctuation
    ep = TemplateWriter("Duck Desk", "Quackers", recent_intros=recent).write(stories)
    assert not [i for i in lint_episode(ep, stories, FRAME, recent_intros=recent) if i.code == "intro_problem"]
