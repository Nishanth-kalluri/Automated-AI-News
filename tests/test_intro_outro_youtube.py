"""The cohesive intro, the daily subscribe line, clean links and credits, and the YouTube upload path."""
import json
import sys
import types
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest

from shorts import cli, pipeline, upload, writer
from shorts.checks import INTRO_WORDS, _speech_problems, intro_shape_problems, lint_episode, unsupported_numbers
from shorts.config import DEFAULT_OUTRO, Config
from shorts.content import clean_url, credit, on_newsletter_host, shared_run
from shorts.models import Episode, Segment, UploadResult
from shorts.outro import SIGN_OFF, SUBSCRIBE_LINES, OutroLog, outro_text
from shorts.upload import (LocalUploader, YouTubeSignInError, YouTubeUploader, upload_hint, upload_metadata,
                           youtube_description, youtube_tags, youtube_title)
from shorts.writer import (CRITIC_SYSTEM, REVIEW_SCHEMA, TEMPLATE_HOOKS, WRITER_SYSTEM, CriticWriter, LLMWriter,
                           TemplateWriter, description_footer)
from tests.test_agents import ScriptedLLM, _story
from tests.test_shadow import _stub_run

HOST, SHOW = "Quackers", "Duck Desk"
# Story 1 of the episode that aired on Sep 29, and the two intros that aired that day.
DOTS = ("At DevDay, OpenAI introduced Dots, assistants powered by GPT-6 Astra that pursue your goals in the "
        "background. They can connect to more than four thousand apps, learn your preferences, and let you inspect "
        "their work. Watch how much control users actually have.")
DOTS_HEADLINE = "OpenAI Introduces Always-On Dots"
AIRED = "Quack! OpenAI’s new Dots can work toward your goals in the background. What can they actually do?"
AIRED_2 = "Quack, AI now handles 26 percent of Anthropic’s research work. What happens if that keeps growing?"
GOOD_INTROS = [
    "Quack quack, it's Quackers on Duck Desk! Later, an OpenAI model breaks its rules in some simulated tests. "
    "First, meet Dots.",
    "Waddle in, it's Quackers! OpenAI had a packed day, from a cheaper GPT to a leaky sandbox, starting with Dots.",
    "Quack! Quackers here on Duck Desk. A cheaper GPT and a sandbox slip are ahead, but first, OpenAI's always-on "
    "Dots.",
    "Duck Desk is back, with Quackers! A cheaper GPT is coming up, but first, OpenAI's new always-on Dots agents.",
    "Quack! It's Quackers on Duck Desk. OpenAI Introduces Always-On Dots, and a cheaper GPT follows.",
]


def _shape(text):
    return intro_shape_problems(text, HOST, SHOW, DOTS, DOTS_HEADLINE)


# --- the intro ---------------------------------------------------------------------------------

def test_the_intros_that_aired_on_sep_29_fail_the_shape_checks():
    problems = _shape(AIRED)
    assert any("doesn't greet the viewer as Quackers" in p for p in problems)
    assert any("ends on a question" in p for p in problems)
    assert any('repeats the first story ("' in p and "background" in p for p in problems)
    problems = _shape(AIRED_2)
    assert any("greet" in p for p in problems) and any("question" in p for p in problems)


@pytest.mark.parametrize("intro", GOOD_INTROS)
def test_greet_tease_hand_over_intros_pass(intro):
    assert _shape(intro) == []


def test_intro_length_bounds_and_questions_inside_are_fine():
    long = "Quack quack, it's Quackers on Duck Desk! " + "OpenAI shipped agents, models, sandboxes and plugins, " * 3
    assert any("keep it to 14-22" in p for p in _shape(long))
    assert any(f"only {len('Quack, Quackers here.'.split())} words" in p for p in _shape("Quack, Quackers here."))
    assert INTRO_WORDS == (10, 26)
    # a question is fine when the intro doesn't end on it
    assert _shape("Quack, it's Quackers on Duck Desk! Cheaper GPT? It's coming up. First, OpenAI's Dots.") == []
    assert any("question" in p for p in _shape("Quack, it's Quackers on Duck Desk! Ready for OpenAI's “Dots?”"))


def test_naming_story_1_is_fine_but_saying_its_facts_is_not():
    assert shared_run("First, OpenAI Introduces Always-On Dots today.", DOTS,
                      skip={"openai", "introduces", "always", "on", "dots"}) == ""
    assert shared_run("Its assistants pursue your goals in the background.", DOTS) == "pursue your goals in the"
    assert shared_run("one two three four", "one two three four") == ""  # shorter than the run
    assert shared_run("the and of a to in", "the and of a to in", skip={"the", "and", "of", "a", "to", "in"}) == ""


def test_lint_runs_the_shape_checks_only_with_the_host_or_show():
    story = _story("OpenAI Introduces Always-On Dots", summary=DOTS)
    story.headline = story.title
    ep = Episode(title="t", description="d", tags=[], stories=[story],
                 segments=[Segment("intro", AIRED), Segment("story", DOTS, DOTS_HEADLINE), Segment("outro", DEFAULT_OUTRO)])
    assert not [i for i in lint_episode(ep, [story], "Duck Desk Quackers") if i.code == "intro_problem"]
    issues = [i for i in lint_episode(ep, [story], "Duck Desk Quackers", host=HOST, show=SHOW)
              if i.code == "intro_problem"]
    assert len(issues) == 1 and "greet" in issues[0].detail and "question" in issues[0].detail


def test_the_writer_prompt_asks_for_greet_tease_hand_over_with_the_real_names():
    system = LLMWriter(ScriptedLLM(), SHOW, HOST).system(8)
    assert "greet, tease, hand over" in system and "it's Quackers on Duck Desk!" in system
    assert "never say its facts, numbers or wording" in system and "No questions" in system
    assert "{host}" not in system and "{show}" not in system
    assert "fresh, playful hook" in WRITER_SYSTEM and "Don't count the stories" in WRITER_SYSTEM


def test_the_critic_judges_whether_the_intro_hangs_together():
    assert "intro_quality" in REVIEW_SCHEMA["properties"] and "intro_quality" in REVIEW_SCHEMA["required"]
    assert "doesn't lead into story 1" in CRITIC_SYSTEM
    stories = [_story("Lab ships agent", summary="The lab shipped an agent that books travel in 3 steps. Users say "
                                                  "where and when, the agent compares flights and asks first."),
               _story("Chip is faster", summary="The new chip is 2 times faster at inference than the old one. "
                                                 "Cloud providers say it makes chatbots cheaper to serve this year.")]
    llm = ScriptedLLM({"intro": [], "outro": [], "segments": [],
                       "intro_quality": ["a greeting followed by an unrelated fact"]})
    w = CriticWriter(llm, llm, SHOW, HOST)
    ep = TemplateWriter(SHOW, HOST).write(stories)
    issues = w.review(ep, stories)
    assert [i.code for i in issues] == ["intro_problem"]
    assert "doesn't work as an opening: a greeting followed by an unrelated fact" in issues[0].detail


class _Day(date):
    today_value = date(2026, 9, 29)

    @classmethod
    def today(cls):
        return cls.today_value


HEADLINES = ["OpenAI Introduces Always-On Dots", "GPT-6.1 Sol Costs Less", "Astra Crosses Limits in Simulations",
             "Codex Gets Cloud Workspaces"]


def _material():
    return " ".join(HEADLINES) + " " + DOTS + " 6.1"


@pytest.mark.parametrize("days", range(7))
def test_the_template_intro_passes_its_own_checks_every_day(monkeypatch, days):
    monkeypatch.setattr(writer, "date", _Day)
    _Day.today_value = date(2026, 9, 29) + timedelta(days=days)
    text = TemplateWriter(SHOW, HOST).intro(4, HEADLINES, first_text=DOTS, first_names=DOTS_HEADLINE,
                                            material=_material()).text
    assert _shape(text) == [] and len(text.split()) <= 22
    assert "Dots" not in text  # never describes story 1
    assert text.split("!")[0].split(".")[0] in {h.format(host=HOST, show=SHOW).split("!")[0].split(".")[0]
                                                for h in TEMPLATE_HOOKS}
    assert "GPT-6.1 Sol Costs Less" in text


def test_the_template_intro_skips_headlines_it_cannot_say(monkeypatch):
    monkeypatch.setattr(writer, "date", _Day)
    _Day.today_value = date(2026, 9, 29)
    t = TemplateWriter(SHOW, HOST)
    kw = dict(first_text=DOTS, first_names=DOTS_HEADLINE, material=_material())
    one = t.intro(1, HEADLINES[:1], **kw).text
    assert one.endswith("Let's get right to the top story.") and _shape(one) == []
    odd = t.intro(3, [HEADLINES[0], "Is GPT-7 Coming Next Month?", "The Rundown Says Codex Wins",
                      "An Extremely Long Headline That Goes On And On Forever Here", "OpenAI Raises 90 Billion"],
                  **kw).text
    assert "GPT-7" not in odd and "Rundown" not in odd and "Forever" not in odd and "90" not in odd
    assert _shape(odd) == []
    every = [h.format(host=HOST, show=SHOW) for h in TEMPLATE_HOOKS]
    still = TemplateWriter(SHOW, HOST, recent_intros=every).intro(4, HEADLINES, **kw).text
    assert _shape(still) == []


def test_template_episode_intro_teases_its_other_stories():
    stories = [_story(h, summary=DOTS if i == 0 else f"{h}. The company shared details and prices today, and "
                                                             "developers can try it now in the cloud.")
               for i, h in enumerate(HEADLINES[:3])]
    for s in stories:
        s.headline = s.title
    ep = TemplateWriter(SHOW, HOST).write(stories)
    intro = ep.segments[0].text
    assert "GPT-6.1 Sol Costs Less" in intro and HOST in intro
    story_issues = [i for i in lint_episode(ep, stories, f"{SHOW} {HOST}", host=HOST, show=SHOW)
                    if i.code == "intro_problem"]
    assert story_issues == []


def test_the_standard_intro_is_built_from_the_final_story_segments():
    good = ("The lab shipped an agent that books your travel in 3 steps. You say where and when, it compares "
            "options and asks before paying. Why it matters: assistants are starting to finish whole errands.")
    fast = ("A new chip runs AI models 2 times faster at inference. Serving chatbots gets cheaper, and cheaper "
            "serving usually means more free features for users soon. Watch for the big clouds to adopt it.")
    a = _story("Lab ships agent", summary="The lab shipped an agent that books travel in 3 steps. Users say where and "
                                          "when they want to go, and it asks for approval before paying.")
    b = _story("Chip is faster", summary="The new chip is 2 times faster at inference than the one it replaces. "
                                         "Cloud providers say faster inference makes chatbots cheaper to serve.")
    for s in (a, b):
        s.headline = s.title
    script = {"title": "AI today", "description": "Two stories.", "tags": ["ai"], "intro": AIRED,
              "segments": [{"headline": "Lab ships agent", "key_fact": "", "text": good},
                           {"headline": "Faster chip lands", "key_fact": "", "text": fast}], "outro": ""}
    llm = ScriptedLLM(script, {"segments": []})
    w = CriticWriter(llm, llm, SHOW, HOST, max_repairs=0)
    ep = w.write([a, b])
    assert "Faster chip lands" in ep.segments[0].text  # the headline that airs, not the story's own
    assert w.report["fallbacks"][0]["part"] == "intro"


# --- the outro ---------------------------------------------------------------------------------

def test_every_subscribe_line_is_short_safe_and_different():
    assert len(SUBSCRIBE_LINES) == 24 and len(set(SUBSCRIBE_LINES)) == 24
    for line in SUBSCRIBE_LINES:
        text = outro_text(line)
        assert "subscrib" in line.lower(), line
        assert 8 <= len(line.split()) <= 18, line
        assert _speech_problems(text, outro=True) == [], line
        assert unsupported_numbers(text, "") == [], line
        assert text.endswith(SIGN_OFF) and "—" not in line
    assert DEFAULT_OUTRO == outro_text(SUBSCRIBE_LINES[0])
    assert "get lost in the storm" not in " ".join(SUBSCRIBE_LINES)  # the user's line was a direction, not a script


def test_outro_log_rotates_through_every_line_before_repeating(tmp_path):
    path = tmp_path / "outros.json"
    seen = []
    for _ in range(len(SUBSCRIBE_LINES) + 2):
        log = OutroLog(path)
        text = log.next()
        seen.append(text)
        log.add(text)
    assert seen[:len(SUBSCRIBE_LINES)] == [outro_text(line) for line in SUBSCRIBE_LINES]
    assert seen[len(SUBSCRIBE_LINES):] == [outro_text(SUBSCRIBE_LINES[0]), outro_text(SUBSCRIBE_LINES[1])]
    assert len(json.loads(path.read_text())) == OutroLog.KEEP


def test_outro_log_survives_bad_files_and_retired_lines(tmp_path):
    path = tmp_path / "outros.json"
    path.write_text("not json")
    assert OutroLog(path).next() == outro_text(SUBSCRIBE_LINES[0])
    path.write_text(json.dumps([{"text": "An old line we retired. " + SIGN_OFF}, 5, {"no": "text"}]))
    assert OutroLog(path).next() == outro_text(SUBSCRIBE_LINES[0])
    path.write_text(json.dumps([{"text": outro_text(SUBSCRIBE_LINES[5])}, {"text": "Custom. " + SIGN_OFF}]))
    assert OutroLog(path).next() == outro_text(SUBSCRIBE_LINES[6])


def test_a_run_picks_todays_subscribe_line_and_records_it(monkeypatch, tmp_path):
    rig = _stub_run(monkeypatch, tmp_path)
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "outros.json").write_text(json.dumps([{"date": "2026-09-29", "text": outro_text(SUBSCRIBE_LINES[3])}]))
    video = pipeline.run(replace(rig.cfg, shadow=False, outro=""), upload=False)
    episode = json.loads((video.parent / "03-episode.json").read_text())
    assert episode["segments"][-1]["text"] == outro_text(SUBSCRIBE_LINES[4])
    logged = json.loads((state / "outros.json").read_text())
    assert [e["text"] for e in logged] == [outro_text(SUBSCRIBE_LINES[3]), outro_text(SUBSCRIBE_LINES[4])]


def test_shorts_outro_fixes_the_outro(monkeypatch, tmp_path):
    monkeypatch.setenv("SHORTS_OUTRO", "Subscribe, little ducklings. " + SIGN_OFF)
    assert Config.from_env().outro == "Subscribe, little ducklings. " + SIGN_OFF
    monkeypatch.delenv("SHORTS_OUTRO")
    assert Config.from_env().outro == ""


def test_workflow_passes_the_new_variables():
    flow = (Path(__file__).parent.parent / ".github" / "workflows" / "daily-short.yml").read_text()
    assert "SHORTS_YOUTUBE_PRIVACY: ${{ vars.SHORTS_YOUTUBE_PRIVACY }}" in flow
    assert "SHORTS_OUTRO: ${{ vars.SHORTS_OUTRO }}" in flow


# --- links and credits -------------------------------------------------------------------------

def test_clean_url_drops_tracking_that_names_a_newsletter():
    assert clean_url("https://www.aisi.gov.uk/blog/gpt?utm_source=tldrai") == "https://www.aisi.gov.uk/blog/gpt"
    assert clean_url("https://x.com/a?id=3&utm_medium=email&ref=therundown") == "https://x.com/a?id=3"
    assert clean_url("https://a.com/b?fbclid=1&mc_cid=2&q=ai#top") == "https://a.com/b?q=ai#top"
    assert clean_url("https://a.com/b?ref=producthunt") == "https://a.com/b?ref=producthunt"
    assert clean_url("https://[broken") == "https://[broken" and clean_url("") == ""


def test_the_credit_on_screen_is_the_real_publisher():
    assert on_newsletter_host("https://tracking.tldrnewsletter.com/CL0/abc")
    assert credit("tracking.tldrnewsletter.com", "https://www.aisi.gov.uk/blog/x") == "aisi.gov.uk"
    assert credit("tracking.tldrnewsletter.com", "https://alignment.openai.com/x") == "OpenAI"
    assert credit("TechCrunch", "https://techcrunch.com/a") == "TechCrunch"
    assert credit("n8n.io", "https://blog.n8n.io/x") == "n8n.io"
    assert credit("Hacker News", "https://news.ycombinator.com/item?id=1") == ""
    assert credit("", "") == ""


def test_description_links_and_segments_carry_no_tracking():
    s = _story("Astra crosses limits", summary="x")
    s.url, s.source = "https://www.aisi.gov.uk/blog/astra?utm_source=tldrai", "tracking.tldrnewsletter.com"
    footer = description_footer([s])
    assert "utm_source" not in footer and "https://www.aisi.gov.uk/blog/astra" in footer
    seg = writer._story_segment(s, "text")
    assert seg.url == "https://www.aisi.gov.uk/blog/astra" and seg.source == "aisi.gov.uk"


# --- YouTube metadata --------------------------------------------------------------------------

def _episode(**kw):
    base = dict(title="OpenAI's always-on agents", description="Two stories.\n\nSources:\n- A: https://a.com/x",
                tags=["AI news", "OpenAI"], segments=[])
    return Episode(**{**base, **kw})


def test_titles_descriptions_and_tags_meet_youtubes_rules():
    ep = _episode(title="GPT-6 > GPT-5 <b>\nwow", tags=["AI, news", "The Rundown", "ai news", "x" * 120, "<tag>"],
                  description="Hi <there>\n\nSources:\n- A: https://x.com/a?utm_source=tldrai&id=2")
    meta = upload_metadata(ep)
    assert "<" not in meta["title"] + meta["description"] and ">" not in meta["title"] + meta["description"]
    assert meta["title"] == "GPT-6 › GPT-5 ‹b› wow #Shorts" and "\n" not in meta["title"]
    assert "https://x.com/a?id=2" in meta["description"] and "Sources:\n- A:" in meta["description"]
    assert meta["tags"] == ["AI", "news", "ai news", "‹tag›"]  # no newsletter name, no 120-character tag
    assert youtube_title(_episode(title="  ")) == "AI News Today #Shorts"
    assert len(youtube_title(_episode(title="x" * 300))) <= 100


def test_long_descriptions_and_many_tags_are_cut_to_fit():
    lines = "\n".join(f"- Story {i}: https://example.com/{'é' * 60}/{i}" for i in range(200))
    desc = youtube_description(_episode(description="Intro.\n\nSources:\n" + lines))
    assert len(desc.encode()) <= upload.DESCRIPTION_MAX_BYTES and desc.startswith("Intro.")
    assert desc.split("\n")[-1].startswith("- Story")  # cut at a line, never mid-link
    tags = youtube_tags(_episode(tags=[f"tag number {i}" for i in range(100)]))
    assert sum(len(t) + 2 for t in tags) + len(tags) - 1 <= upload.TAGS_MAX_CHARS and len(tags) > 5


def test_local_uploader_writes_clean_metadata(tmp_path):
    video = tmp_path / "short.mp4"
    video.write_bytes(b"x")
    result = LocalUploader().upload(video, _episode(description="D https://a.com/x?utm_source=tldrai"))
    meta = json.loads(Path(result.location).read_text())
    assert meta["file"] == "short.mp4" and meta["title"].endswith("#Shorts") and "utm" not in meta["description"]


# --- YouTube sign-in and upload ----------------------------------------------------------------

def _secrets(monkeypatch, **values):
    for name in upload.YT_SECRETS:
        monkeypatch.setenv(name, values.get(name, f" {name.lower()}-value\n"))


def test_missing_secrets_are_named(monkeypatch):
    for name in upload.YT_SECRETS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("YOUTUBE_CLIENT_ID", "id")
    with pytest.raises(YouTubeSignInError, match="YOUTUBE_CLIENT_SECRET, YOUTUBE_REFRESH_TOKEN"):
        YouTubeUploader("private")
    monkeypatch.setenv("YOUTUBE_CLIENT_SECRET", "   ")
    with pytest.raises(YouTubeSignInError, match="YOUTUBE_CLIENT_SECRET"):
        YouTubeUploader("private")


def test_secrets_are_read_without_stray_whitespace(monkeypatch):
    _secrets(monkeypatch)
    creds = YouTubeUploader("private").credentials()
    assert creds.refresh_token == "youtube_refresh_token-value" and creds.client_id == "youtube_client_id-value"


def test_a_refused_sign_in_says_what_to_do(monkeypatch):
    from google.auth.exceptions import RefreshError
    from google.oauth2.credentials import Credentials

    _secrets(monkeypatch)

    def refuse(self, request):
        raise RefreshError("invalid_grant: Token has been expired or revoked.")

    monkeypatch.setattr(Credentials, "refresh", refuse)
    with pytest.raises(YouTubeSignInError) as caught:
        YouTubeUploader("private").check()
    hint = upload_hint(caught.value)
    assert "youtube-auth" in hint and "YOUTUBE_REFRESH_TOKEN" in hint and "7 days" in hint
    assert "needs the secret" not in hint
    assert "Secrets and variables" in upload_hint(YouTubeSignInError("YouTube upload needs the secret X"))
    assert "daily upload limit" in upload_hint(RuntimeError("<HttpError 403 ... quotaExceeded>"))
    assert upload_hint(ValueError("boom")) == "ValueError: boom"


class _Request:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.retries = []

    def next_chunk(self, num_retries=0):
        self.retries.append(num_retries)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return None, outcome


def _fake_google(monkeypatch, request):
    calls = {}

    class Videos:
        def insert(self, **kw):
            calls["insert"] = kw
            return request

    class YouTube:
        def videos(self):
            return Videos()

    discovery = types.ModuleType("googleapiclient.discovery")
    discovery.build = lambda *a, **kw: calls.setdefault("build", (a, kw)) and YouTube()
    http = types.ModuleType("googleapiclient.http")
    http.MediaFileUpload = lambda path, **kw: calls.setdefault("media", (path, kw))
    monkeypatch.setitem(sys.modules, "googleapiclient.discovery", discovery)
    monkeypatch.setitem(sys.modules, "googleapiclient.http", http)
    monkeypatch.setattr(upload.time, "sleep", lambda s: None)
    return calls


def test_upload_sends_clean_metadata_and_resumes_after_a_dropped_connection(monkeypatch, tmp_path):
    _secrets(monkeypatch)
    request = _Request([ConnectionResetError("reset"), None, {"id": "abc123"}])
    calls = _fake_google(monkeypatch, request)
    video = tmp_path / "short.mp4"
    video.write_bytes(b"x")
    result = YouTubeUploader("private").upload(video, _episode(description="D https://a.com/x?utm_source=tldrai"))
    assert result == UploadResult("youtube", "https://youtube.com/shorts/abc123")
    body = calls["insert"]["body"]
    assert body["snippet"]["defaultLanguage"] == "en" and body["snippet"]["defaultAudioLanguage"] == "en"
    assert body["snippet"]["title"].endswith("#Shorts") and "utm" not in body["snippet"]["description"]
    assert body["status"] == {"privacyStatus": "private", "selfDeclaredMadeForKids": False,
                              "containsSyntheticMedia": True}
    assert request.retries == [0, 0, 0]  # never the client's own retries: they resend an empty chunk
    assert json.loads((tmp_path / "upload.json").read_text())["file"] == "short.mp4"  # for a hand upload too
    pasted = (tmp_path / "upload.txt").read_text(encoding="utf-8")
    assert pasted.startswith("TITLE\nOpenAI's always-on agents #Shorts\n\nDESCRIPTION\nD https://a.com/x\n")
    assert pasted.endswith("TAGS\nAI news, OpenAI\n")


def _http_error(status):
    from googleapiclient.errors import HttpError

    return HttpError(types.SimpleNamespace(status=status, reason="x"), b"{}")


def test_upload_resumes_after_youtube_is_busy_but_not_after_a_rejection(monkeypatch, tmp_path):
    _secrets(monkeypatch)
    video = tmp_path / "short.mp4"
    video.write_bytes(b"x")
    request = _Request([_http_error(503), _http_error(429), {"id": "ok"}])
    _fake_google(monkeypatch, request)
    assert YouTubeUploader("private").upload(video, _episode()).location.endswith("/ok")
    request = _Request([_http_error(400), {"id": "never"}])
    _fake_google(monkeypatch, request)
    with pytest.raises(Exception) as caught:
        YouTubeUploader("private").upload(video, _episode())
    assert caught.value.resp.status == 400 and len(request.retries) == 1


def test_upload_gives_up_after_repeated_drops(monkeypatch, tmp_path):
    _secrets(monkeypatch)
    _fake_google(monkeypatch, _Request([OSError("down")] * (upload.UPLOAD_RETRIES + 1)))
    video = tmp_path / "short.mp4"
    video.write_bytes(b"x")
    with pytest.raises(OSError):
        YouTubeUploader("public").upload(video, _episode())


def test_privacy_is_checked_when_the_run_starts(monkeypatch):
    monkeypatch.setenv("SHORTS_YOUTUBE_PRIVACY", "Public")
    assert Config.from_env().youtube_privacy == "public"
    monkeypatch.setenv("SHORTS_YOUTUBE_PRIVACY", "publik")
    with pytest.raises(ValueError, match="private, unlisted, public"):
        Config.from_env()
    monkeypatch.delenv("SHORTS_YOUTUBE_PRIVACY")
    monkeypatch.setenv("SHORTS_UPLOADER", "YouTube")
    assert Config.from_env().uploader == "youtube" and Config.from_env().youtube_privacy == "private"


# --- the run: sign in first, never lose the episode, say where it went ---------------------------

class _Refused:
    name = "youtube"

    def check(self):
        raise YouTubeSignInError("Google refused the saved YouTube sign-in (invalid_grant)")

    def upload(self, video, episode):  # pragma: no cover - must never be called
        raise AssertionError("uploaded after a refused sign-in")


def test_a_refused_sign_in_still_makes_the_episode_for_a_hand_upload(monkeypatch, tmp_path):
    rig = _stub_run(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "build_uploader", lambda cfg: _Refused())
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", "me/show")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    video = pipeline.run(replace(rig.cfg, shadow=False), upload=True)
    assert (video.parent / "upload.json").exists()
    subject, text = rig.notes[-1]
    assert subject.startswith("New episode ready, NOT uploaded: ")
    assert text.startswith("NOT uploaded to YouTube: Google refused") and "youtube-auth" in text
    assert "https://github.com/me/show/actions/runs/42" in text


def test_a_failed_upload_keeps_the_episode_and_says_why(monkeypatch, tmp_path):
    rig = _stub_run(monkeypatch, tmp_path)

    class Broken:
        name = "youtube"

        def upload(self, video, episode):
            raise RuntimeError("<HttpError 403 ... quotaExceeded>")

    monkeypatch.setattr(pipeline, "build_uploader", lambda cfg: Broken())
    video = pipeline.run(replace(rig.cfg, shadow=False), upload=True)
    assert (video.parent / "upload.json").exists()
    subject, text = rig.notes[-1]
    assert "NOT uploaded" in subject and "daily upload limit" in text
    assert json.loads((tmp_path / "state" / "outros.json").read_text())  # the episode still counts as made


def test_the_email_says_a_private_upload_waits_on_the_api_audit(monkeypatch, tmp_path):
    rig = _stub_run(monkeypatch, tmp_path)
    pipeline.run(replace(rig.cfg, shadow=False), upload=True)
    subject, text = rig.notes[-1]
    assert subject.startswith("New episode ready: ")
    assert text.startswith("Uploaded to YouTube as private: https://youtube.com/shorts/abc")
    assert "API audit" in text and "SHORTS_YOUTUBE_PRIVACY" in text and "make it public in YouTube Studio" not in text
    rig.notes.clear()
    (tmp_path / "state" / "seen_urls.json").unlink()  # the same news again
    pipeline.run(replace(rig.cfg, shadow=False, youtube_privacy="public"), upload=True)
    assert rig.notes[-1][1].startswith("Uploaded to YouTube as public:") and "API audit" not in rig.notes[-1][1]
    rig.notes.clear()
    (tmp_path / "state" / "seen_urls.json").unlink()
    pipeline.run(replace(rig.cfg, shadow=False), upload=False)
    assert rig.notes[-1][1].startswith("Not uploaded (this run was started without upload).")


def test_youtube_auth_prints_all_three_secrets(monkeypatch, capsys):
    monkeypatch.setattr(upload, "youtube_auth_flow", lambda path: {
        "YOUTUBE_CLIENT_ID": "cid", "YOUTUBE_CLIENT_SECRET": "sec", "YOUTUBE_REFRESH_TOKEN": "tok"})
    assert cli.main(["youtube-auth", "client_secret.json"]) == 0
    out = capsys.readouterr().out
    assert "YOUTUBE_CLIENT_ID=cid\nYOUTUBE_CLIENT_SECRET=sec\nYOUTUBE_REFRESH_TOKEN=tok" in out
    monkeypatch.setattr(upload, "youtube_auth_flow", lambda path: {
        "YOUTUBE_CLIENT_ID": "cid", "YOUTUBE_CLIENT_SECRET": "sec", "YOUTUBE_REFRESH_TOKEN": ""})
    assert cli.main(["youtube-auth", "client_secret.json"]) == 1


def test_client_ids_come_from_the_downloaded_json(tmp_path):
    path = tmp_path / "client_secret.json"
    path.write_text(json.dumps({"installed": {"client_id": "abc.apps.googleusercontent.com", "client_secret": "s"}}))
    assert upload._client_ids(str(path)) == ("abc.apps.googleusercontent.com", "s")


# --- review fixes: greetings with possessives, stock phrases, story 1 dropped, rewrites, retries ---

def test_greetings_with_a_possessive_or_contraction_count():
    for intro in ("Quack, Duck Desk's on the air! A cheaper GPT is coming up. First, OpenAI's Dots.",
                  "Duck Desk’s back with the pond news! A cheaper GPT is ahead, but first, OpenAI's Dots.",
                  "Quackers'll walk you through it! A cheaper GPT is ahead, but first, OpenAI's Dots."):
        assert _shape(intro) == [], intro
    assert any("greet" in p for p in _shape("Quack! Desk news today, a cheaper GPT is ahead. First, OpenAI's Dots."))


def test_a_stock_phrase_another_story_shares_is_not_a_repeat_of_story_1():
    first = "Google made Gemini 3 Flash the default in its app, and it rolls out in the coming weeks to everyone."
    other = "Microsoft says Copilot's memory rolls out in the coming weeks for Plus and Pro users."
    intro = "Quack! Quackers here on Duck Desk. Copilot's memory rolls out in the coming weeks. First, Gemini."
    assert any("repeats the first story" in p for p in intro_shape_problems(intro, HOST, SHOW, first, "Gemini"))
    assert intro_shape_problems(intro, HOST, SHOW, first, "Gemini", other) == []


def test_saying_story_1s_opening_sentence_is_caught_even_in_headline_words():
    first = "Google made Gemini 3 Flash the default model in the Gemini app. Users get faster answers from today."
    intro = "Quack quack, it's Quackers on Duck Desk! Google made Gemini 3 Flash the default model in the Gemini app."
    problems = intro_shape_problems(intro, HOST, SHOW, first, "Google makes Gemini 3 Flash the default model")
    assert any("repeats the first story" in p and "Gemini 3 Flash" in p for p in problems)


def test_when_story_1_is_left_out_the_intro_is_rebuilt_for_the_new_first_story():
    from tests.test_review_fixes import GOOD_A, GOOD_B, NO_ISSUES, _lazy_story, _script, _stories

    stories = [_lazy_story(), *_stories()]
    forum = ("Over on Reddit, users say GPT-6 now gives shorter answers than GPT-5 did. OpenAI says it is looking "
             "into the reports and will share an update soon. People notice quickly when a model changes.")
    intro = "Quack, it's Quackers on Duck Desk! A faster chip is coming up. But first, the big one."
    llm = ScriptedLLM(_script(forum, GOOD_A, GOOD_B, intro=intro), NO_ISSUES)
    w = CriticWriter(llm, llm, SHOW, HOST, max_repairs=0)
    ep = w.write(stories)
    assert [d["story"] for d in w.report["dropped"]] == [1]
    assert ep.segments[0].text != intro
    assert ep.segments[0].text == w.template.intro_for(ep.stories, ep.story_segments).text
    assert {"part": "intro", "why": "led into the first story, which was left out"} in w.report["fallbacks"]


def test_an_intro_rewrite_is_shown_the_recent_intros():
    from tests.test_review_fixes import GOOD_A, GOOD_B, NO_CHANGE, NO_ISSUES, _script, _stories

    recent = ["Quack quack, it's Quackers on Duck Desk! Big news from OpenAI today."]
    script = _script(GOOD_A, GOOD_B, intro="Quack! A faster chip is coming up. First, a lab agent for trips.")
    llm = ScriptedLLM(script, NO_ISSUES, NO_CHANGE, NO_ISSUES)
    CriticWriter(llm, llm, SHOW, HOST, max_repairs=1, recent_intros=recent).write(_stories())
    assert "Recent intros (greet and tease differently from all of these):\n- " + recent[0] in llm.calls[2][1]
    llm = ScriptedLLM(_script(GOOD_A.replace("The lab", "The revolutionary lab"), GOOD_B), NO_ISSUES, NO_CHANGE,
                      NO_ISSUES)
    CriticWriter(llm, llm, SHOW, HOST, max_repairs=1, recent_intros=recent).write(_stories())
    assert "Recent intros" not in llm.calls[2][1]  # only when the intro is being rewritten


def test_no_outro_sentence_is_a_phrase_a_story_could_say():
    from shorts.content import _sentences

    for line in SUBSCRIBE_LINES:
        for sentence in _sentences(line):
            assert "subscrib" in sentence.lower() or len(sentence.split()) >= 5 or len(sentence.split()) < 3, sentence


def test_a_network_blip_at_sign_in_does_not_cancel_the_upload(monkeypatch):
    from google.auth.exceptions import RefreshError, TransportError
    from google.oauth2.credentials import Credentials

    _secrets(monkeypatch)
    for exc in (TransportError("connection reset"), RefreshError("503 from Google", retryable=True)):
        def fail(self, request, exc=exc):
            raise exc

        monkeypatch.setattr(Credentials, "refresh", fail)
        YouTubeUploader("private").check()  # logs and leaves it to the upload

    class Flaky:
        name = "youtube"

        def check(self):
            raise RuntimeError("something odd")

    monkeypatch.setattr(pipeline, "build_uploader", lambda cfg: Flaky())
    cfg = replace(Config.from_env(), uploader="youtube")
    uploader, problem = pipeline._uploader(cfg, True)
    assert isinstance(uploader, Flaky) and problem == ""
    monkeypatch.setattr(pipeline, "build_uploader", lambda cfg: _Refused())
    uploader, problem = pipeline._uploader(cfg, True)
    assert isinstance(uploader, LocalUploader) and "youtube-auth" in problem
    assert pipeline._uploader(cfg, False)[1] == "" and pipeline._uploader(replace(cfg, uploader="local"), True)[1] == ""
