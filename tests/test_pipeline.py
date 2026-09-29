import json
import wave
from datetime import datetime, timedelta, timezone

from shorts import research, sources
from shorts.character import loudness_envelope, mouth_states
from shorts.composer import _ass_time, caption_chunks, captions_ass
from shorts.config import Config
from shorts.media import media_duration
from shorts.models import Story, Word
from shorts.pipeline import run
from shorts.selection import LLMEditor, SeenStore, pick_stories, pick_with_fallback
from shorts.writer import LLMWriter, TemplateWriter, write_episode


def _story(title, hours_ago=1, url=None, kind="article", summary="An AI model from OpenAI"):
    return Story(title=title, url=url or f"https://x/{title}", source="t",
                 published=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
                 summary=summary, kind=kind)


class FakeLLM:
    name, model = "fake", "fake-1"

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def json(self, system, user, *, stage, schema=None):
        self.prompts.append((stage, user))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class FakeResponse:
    def __init__(self, data, url=""):
        self.data, self.url, self.status_code, self.text = data, url, 200, json.dumps(data)

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


def test_pick_dedupes_and_skips_seen_stale_and_newsletters():
    stories = [
        _story("OpenAI releases new model"),
        _story("OpenAI releases a new model"),  # near-duplicate
        _story("Anthropic ships Claude agent", url="https://seen"),
        _story("Old AI news", hours_ago=100),
        _story("Nvidia AI chip"),
        _story("The Rundown AI issue", kind="newsletter"),
    ]
    titles = [s.title for s in pick_stories(stories, 3, max_age_hours=36, seen={"https://seen"})]
    assert len(titles) == 2 and "Nvidia AI chip" in titles  # one of the two OpenAI duplicates, nothing else


def test_html_to_text_keeps_links():
    text = sources.html_to_text('<style>x{}</style><p>OpenAI shipped <a href="https://o.ai/a">a model</a>.</p>')
    assert text == "OpenAI shipped a model (https://o.ai/a)."


def test_newsletter_source_reads_recent_issues(monkeypatch):
    long_html = "<p>" + "Big AI news today. " * 40 + '<a href="https://example.org/story">Read</a></p>'
    now = datetime.now(timezone.utc)
    recent, old = (f"{now - timedelta(hours=h):%Y-%m-%dT%H:%M:%SZ}" for h in (2, 72))
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        assert headers["Authorization"] == "Bearer key"
        if url.endswith("/messages"):
            return FakeResponse({"messages": [{"message_id": "m1", "timestamp": recent},
                                              {"message_id": "m2", "timestamp": recent},
                                              {"message_id": "m3", "timestamp": old}]})
        if url.endswith("/m1"):
            return FakeResponse({"subject": "Today in AI", "from": "The Rundown AI <news@rundown.ai>",
                                 "timestamp": recent, "html": long_html})
        if url.endswith("/m3"):
            return FakeResponse({"subject": "Last week in AI", "from": "The Rundown AI <news@rundown.ai>",
                                 "timestamp": old, "html": long_html})
        return FakeResponse({"subject": "Confirm your subscription", "text": "Click to confirm."})

    monkeypatch.setattr(sources.requests, "get", fake_get)
    got = sources.NewsletterSource("key", "news@agentmail.to", 30).fetch()
    assert calls[0] == "https://api.agentmail.to/v0/inboxes/news@agentmail.to/messages"
    assert not any(c.endswith("/m3") for c in calls)  # older than 30 hours: skipped from the listing
    assert len(got) == 1  # the short confirmation mail is skipped
    issue = got[0]
    assert (issue.kind, issue.source, issue.title) == ("newsletter", "The Rundown AI", "Today in AI")
    assert "(https://example.org/story)" in issue.body


def test_newsletter_source_without_key_is_empty():
    assert sources.NewsletterSource("", "", 30).fetch() == []


def test_llm_editor_reads_newsletters_and_maps_picks(tmp_path):
    summary = ("The lab's new agent books flights and hotels end to end. It asks the user to confirm "
               "before paying and finishes a booking in 3 steps.")
    reply = {"stories": [{"headline": "Lab ships agent", "summary": summary, "key_fact": "3 steps",
                          "url": "https://lab.ai/agent", "outlets": ["TLDR AI", "The Neuron"], "why": "big"}]}
    llm = FakeLLM(reply)
    issue = _story("TLDR AI 2026-09-26", kind="newsletter")
    issue.source, issue.body = "TLDR AI", "NEWSLETTER BODY TEXT"
    seen = SeenStore(tmp_path / "seen.json")
    picked = LLMEditor(llm, 30).pick([issue, _story("Feed headline")], 1, seen)
    assert "NEWSLETTER BODY TEXT" in llm.prompts[0][1] and "Feed headline" in llm.prompts[0][1]
    s = picked[0]
    # the on-screen credit is the link's publisher, never the newsletter; the newsletters stay in outlets
    assert (s.headline, s.url, s.source, s.key_fact) == ("Lab ships agent", "https://lab.ai/agent", "lab.ai", "3 steps")
    assert s.outlets == ["TLDR AI", "The Neuron"] and s.summary == summary


def test_editor_falls_back_to_heuristic_when_llm_fails(tmp_path):
    seen = SeenStore(tmp_path / "seen.json")
    candidates = [
        _story("OpenAI model", summary="OpenAI released a smaller reasoning model for developers. It costs half as "
                                       "much per token and matches the larger model on coding benchmarks."),
        _story("Nvidia AI chip", summary="Nvidia unveiled a data center chip built for inference. The company says "
                                         "it serves language models three times faster using the same power."),
        _story("Google AI agent news", summary="Google AI agent news today."),  # nothing beyond the headline
    ]
    picked = pick_with_fallback(LLMEditor(FakeLLM(RuntimeError("down")), 30), candidates, 3, seen, 30)
    assert len(picked) == 2 and picked[0].headline  # two solid stories rather than a thin third one
    assert "Google AI agent news" not in [s.title for s in picked]


def test_seen_store_reads_old_url_list(tmp_path):
    path = tmp_path / "seen.json"
    path.write_text(json.dumps(["https://old"]))
    seen = SeenStore(path)
    assert seen.urls == {"https://old"}
    s = _story("New thing")
    s.headline = "New thing"
    seen.add([s])
    assert "New thing" in SeenStore(path).recent_headlines()


def test_llm_writer_builds_episode_and_template_fallback():
    stories = [_story("A"), _story("B")]
    for s in stories:
        s.headline = s.title
    good = {"title": "T", "description": "D", "tags": ["ai"], "intro": "Hi there.",
            "segments": [{"text": "Story A text."}, {"text": "Story B text.", "key_fact": "2x"}], "outro": "Bye."}
    ep = LLMWriter(FakeLLM(good), "Show", "Host").write(stories)
    assert [s.kind for s in ep.segments] == ["intro", "story", "story", "outro"]
    assert ep.segments[2].key_fact == "2x" and "https://x/A" in ep.description
    bad = dict(good, segments=[{"text": "only one"}])
    ep = write_episode(LLMWriter(FakeLLM(bad), "Show", "Host"), stories, "Show", "Host")
    assert len(ep.story_segments) == 2 and ep.title.startswith("AI News Today")
    assert isinstance(TemplateWriter("S", "H").write(stories).segments[0].text, str)


def test_tavily_researcher_fills_article_text(monkeypatch):
    s = _story("Agent", url="https://t.co/abc")
    monkeypatch.setattr(research, "resolve_url", lambda url: "https://lab.ai/agent")

    def fake_post(url, json=None, headers=None, timeout=None):
        assert json["urls"] == ["https://lab.ai/agent"]
        return FakeResponse({"results": [{"url": "https://lab.ai/agent", "raw_content": "Full article"}],
                             "failed_results": []})

    monkeypatch.setattr(research.requests, "post", fake_post)
    research.TavilyResearcher("tvly").enrich([s])
    assert (s.url, s.body) == ("https://lab.ai/agent", "Full article")


def test_captions_break_at_sentences_and_time_format():
    words = [Word(w, i * 0.5, i * 0.5 + 0.5) for i, w in enumerate("One two. Three four five six".split())]
    assert [[w.text for w in c] for c in caption_chunks(words)] == [["One", "two."], ["Three", "four", "five"], ["six"]]
    assert _ass_time(3725.5) == "1:02:05.50"
    assert "{\\k50}One {\\k50}two." in captions_ass(words)


def test_mouth_follows_loudness(tmp_path):
    rate, path = 16000, tmp_path / "tone.wav"
    import math
    frames = b"".join(
        int((8000 if (i // rate) % 2 else 0) * math.sin(i / 8)).to_bytes(2, "little", signed=True)
        for i in range(rate * 2))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate), w.writeframes(frames)
    states = mouth_states(loudness_envelope(path))
    assert set(states[:10]) == {0} and max(states[20:]) == 2


def test_offline_run_produces_two_minute_style_episode(tmp_path, monkeypatch):
    monkeypatch.setenv("SHORTS_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("SHORTS_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("SHORTS_X264_PRESET", "ultrafast")
    video = run(Config.from_env().offline())
    assert video.exists() and video.stat().st_size > 100_000
    assert 45 < media_duration(video) <= 180
    folder = video.parent
    for name in ("01-candidates.json", "02-picks.json", "03-episode.json", "timeline.json", "captions.ass",
                 "qa.json", "upload.json", "cost.json"):
        assert (folder / name).exists(), name
    episode = json.loads((folder / "03-episode.json").read_text())
    assert len([s for s in episode["segments"] if s["kind"] == "story"]) == 8
