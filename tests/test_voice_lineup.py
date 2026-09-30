import json
import subprocess
import sys
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import openai
import pytest

from shorts import cli, voice
from shorts.config import DEFAULT_OPENAI_TTS_INSTRUCTIONS, DEFAULT_OUTRO, Config
from shorts.media import media_duration, run_ffmpeg
from shorts.models import Episode, Segment
from shorts.voice import LINEUP_EDGE, LINEUP_OPENAI, OpenAIVoice, SilentVoice, build_voice, lineup, parse_lineup
from shorts.writer import episode_json

REPO = Path(__file__).resolve().parents[1]
SIGN_OFF = "That's the news from the pond. See you tomorrow!"
VOICE_ENV = ("SHORTS_EDGE_RATE", "SHORTS_VOICE", "SHORTS_EDGE_VOICE", "SHORTS_OPENAI_VOICE", "SHORTS_OPENAI_TTS_MODEL",
             "SHORTS_OPENAI_TTS_INSTRUCTIONS", "SHORTS_MIN_STORIES", "SHORTS_ALLOW_NO_AI", "SHORTS_OUTRO",
             "SHORTS_VOICE_LINEUP", "SHORTS_SOURCES", "SHORTS_HOST_NAME")
CLIP_SECONDS = 0.4


def _clear_env(monkeypatch):
    for name in VOICE_ENV:
        monkeypatch.delenv(name, raising=False)


def _cfg(**kw):
    base = dict(edge_rate="+18%", openai_api_key="", host_name="Quackers", openai_tts_model="gpt-4o-mini-tts",
                openai_tts_instructions=DEFAULT_OPENAI_TTS_INSTRUCTIONS, voice_lineup=[])
    return replace(Config.from_env(), **{**base, **kw})


def _episode():
    return Episode(title="Duck Desk", description="Today's AI news.", tags=["ai"], segments=[
        Segment("intro", "Quack quack, it's Quackers on Duck Desk."),
        Segment("story", "OpenAI shipped a new model to ChatGPT users.", headline="OpenAI ships a model"),
        Segment("outro", DEFAULT_OUTRO),
    ])


def _wav(path, seconds=CLIP_SECONDS):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(voice.SAMPLE_RATE)
        w.writeframes(b"\x00\x00" * int(voice.SAMPLE_RATE * seconds))
    return path


class FakeVoice:
    """A working TTS stand-in (not a SilentVoice, so a failure falls back to silence like a real voice)."""

    def __init__(self, provider, name, fail_on="", garbage=False):
        self.name, self.voice, self.fail_on, self.garbage, self.spoken = provider, name, fail_on, garbage, []

    def speak(self, text, out_path):
        if self.fail_on and self.fail_on in text:
            raise RuntimeError("tts service down")
        self.spoken.append(text)
        if self.garbage:
            path = out_path.with_suffix(".mp3")
            path.write_bytes(b"not audio at all")
            return path
        return _wav(out_path.with_suffix(".wav"))


def _fake_edge(monkeypatch, **kw):
    made = []

    def factory(name, rate):
        v = FakeVoice("edge", name, **kw)
        v.rate = rate
        made.append(v)
        return v

    monkeypatch.setattr(voice, "EdgeVoice", factory)
    return made


class FakeOpenAIClient:
    calls: list = []
    content = b""

    def __init__(self, **kw):
        self.kw = kw
        self.audio = SimpleNamespace(speech=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        FakeOpenAIClient.calls.append(kw)
        return SimpleNamespace(content=FakeOpenAIClient.content)


@pytest.fixture
def fake_openai(monkeypatch, tmp_path):
    mp3 = tmp_path / "tone.mp3"
    run_ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", str(CLIP_SECONDS), "-codec:a", "libmp3lame",
                str(mp3)])
    monkeypatch.setattr(FakeOpenAIClient, "calls", [])
    monkeypatch.setattr(FakeOpenAIClient, "content", mp3.read_bytes())
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAIClient)
    return FakeOpenAIClient


def _work_dirs(out_dir):
    return [p.name for p in out_dir.iterdir() if p.is_dir() or p.name.startswith(".")]


# --- config defaults ---------------------------------------------------------------------------------------------


def test_config_defaults_are_the_faster_edge_voice_and_the_fixed_outro(monkeypatch):
    _clear_env(monkeypatch)
    cfg = Config.from_env()
    assert (cfg.voice, cfg.edge_voice, cfg.edge_rate) == ("edge", "en-US-AnaNeural", "+18%")
    assert (cfg.openai_voice, cfg.openai_tts_model) == ("coral", "gpt-4o-mini-tts")
    assert "{host}" in cfg.openai_tts_instructions
    assert cfg.min_stories == 4
    assert cfg.allow_no_ai is False
    assert cfg.voice_lineup == []
    assert cfg.sources == ["newsletter", "rss"]
    assert cfg.outro == ""  # the daily rotation
    assert DEFAULT_OUTRO.endswith(SIGN_OFF)
    assert "subscribe" in DEFAULT_OUTRO.lower()


def test_config_voice_settings_follow_the_environment(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("SHORTS_EDGE_RATE", "+25%")
    monkeypatch.setenv("SHORTS_VOICE", "openai")
    monkeypatch.setenv("SHORTS_EDGE_VOICE", "en-GB-MaisieNeural")
    monkeypatch.setenv("SHORTS_OPENAI_VOICE", "nova")
    monkeypatch.setenv("SHORTS_OPENAI_TTS_MODEL", "tts-1")
    monkeypatch.setenv("SHORTS_OPENAI_TTS_INSTRUCTIONS", "Talk like {host}.")
    monkeypatch.setenv("SHORTS_MIN_STORIES", "3")
    monkeypatch.setenv("SHORTS_ALLOW_NO_AI", "on")
    monkeypatch.setenv("SHORTS_OUTRO", "Bye from the pond!")
    monkeypatch.setenv("SHORTS_VOICE_LINEUP", "edge, openai:coral ,")
    cfg = Config.from_env()
    assert (cfg.edge_rate, cfg.voice, cfg.edge_voice, cfg.openai_voice) == ("+25%", "openai", "en-GB-MaisieNeural",
                                                                            "nova")
    assert (cfg.openai_tts_model, cfg.openai_tts_instructions) == ("tts-1", "Talk like {host}.")
    assert (cfg.min_stories, cfg.allow_no_ai, cfg.outro) == (3, True, "Bye from the pond!")
    assert cfg.voice_lineup == ["edge", "openai:coral"]
    monkeypatch.setenv("SHORTS_ALLOW_NO_AI", "false")
    assert Config.from_env().allow_no_ai is False


def test_offline_config_reads_no_voice_lineup(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("SHORTS_VOICE_LINEUP", "all")
    cfg = Config.from_env()
    assert cfg.voice_lineup == ["all"]
    off = cfg.offline()
    assert (off.voice_lineup, off.voice) == ([], "silent")


# --- parse_lineup --------------------------------------------------------------------------------------------------


def test_parse_lineup_all_is_the_edge_then_openai_set_without_duplicates():
    everything = [("edge", v) for v in LINEUP_EDGE] + [("openai", v) for v in LINEUP_OPENAI]
    assert parse_lineup(["all"]) == everything
    assert len(everything) == len(set(everything)) == 14
    assert everything[0] == ("edge", "en-US-AnaNeural")
    assert parse_lineup(["all", "edge", "openai", " ALL "]) == everything


def test_parse_lineup_edge_and_openai_halves():
    assert parse_lineup(["edge"]) == [("edge", v) for v in LINEUP_EDGE]
    assert parse_lineup(["openai"]) == [("openai", v) for v in LINEUP_OPENAI]
    assert {p for p, _ in parse_lineup(["edge"])} == {"edge"}
    assert ("openai", "coral") in parse_lineup(["openai"])


def test_parse_lineup_explicit_and_bare_names():
    assert parse_lineup(["edge:en-GB-MaisieNeural"]) == [("edge", "en-GB-MaisieNeural")]
    assert parse_lineup(["openai:coral"]) == [("openai", "coral")]
    assert parse_lineup(["en-US-GuyNeural"]) == [("edge", "en-US-GuyNeural")]
    assert parse_lineup(["nova"]) == [("openai", "nova")]
    assert parse_lineup([" OpenAI : shimmer "]) == [("openai", "shimmer")]


def test_parse_lineup_ignores_blanks_and_keeps_order_without_duplicates():
    assert parse_lineup(["", "   ", "edge:"]) == []
    specs = ["openai:nova", "", "en-US-GuyNeural", "nova", "edge:en-US-GuyNeural", "openai:coral", "openai:nova"]
    assert parse_lineup(specs) == [("openai", "nova"), ("edge", "en-US-GuyNeural"), ("openai", "coral")]


# --- OpenAI voice and build_voice ---------------------------------------------------------------------------------


def test_openai_voice_sends_the_settings_and_writes_an_mp3(tmp_path):
    speech = SimpleNamespace(calls=[])
    speech.create = lambda **kw: speech.calls.append(kw) or SimpleNamespace(content=b"ID3-fake-mp3")
    client = SimpleNamespace(audio=SimpleNamespace(speech=speech))
    v = OpenAIVoice("sk-test", "coral", "gpt-4o-mini-tts", "Be a cheerful duck.", client=client)
    path = v.speak("Quack quack, it's Quackers.", tmp_path / "seg_00")
    assert v.name == "openai"
    assert path == tmp_path / "seg_00.mp3"
    assert path.read_bytes() == b"ID3-fake-mp3"
    assert speech.calls == [{"model": "gpt-4o-mini-tts", "voice": "coral", "input": "Quack quack, it's Quackers.",
                             "response_format": "mp3", "instructions": "Be a cheerful duck."}]


def test_openai_voice_without_instructions_leaves_them_out(tmp_path):
    speech = SimpleNamespace(calls=[])
    speech.create = lambda **kw: speech.calls.append(kw) or SimpleNamespace(content=b"x")
    v = OpenAIVoice("sk-test", "nova", "tts-1", client=SimpleNamespace(audio=SimpleNamespace(speech=speech)))
    v.speak("Hello pond.", tmp_path / "seg_01.wav")
    assert "instructions" not in speech.calls[0]
    assert (tmp_path / "seg_01.mp3").read_bytes() == b"x"


def test_openai_voice_builds_its_client_from_the_key(monkeypatch):
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAIClient)
    v = OpenAIVoice("sk-test", "coral", "gpt-4o-mini-tts")
    assert isinstance(v.client, FakeOpenAIClient) and v.client.kw == {"api_key": "sk-test"}


def test_build_voice_openai_needs_a_key():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_voice(_cfg(voice="openai", openai_api_key=""))


def test_build_voice_openai_fills_in_the_host_name(monkeypatch):
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAIClient)
    v = build_voice(_cfg(voice="openai", openai_api_key="sk-test", openai_voice="nova", host_name="Waddles"))
    assert isinstance(v, OpenAIVoice)
    assert (v.voice, v.model, v.client.kw) == ("nova", "gpt-4o-mini-tts", {"api_key": "sk-test"})
    assert v.instructions.startswith("You are Waddles, ")
    assert "{host}" not in v.instructions


def test_build_voice_edge_uses_the_configured_rate_and_rejects_unknown_providers():
    pytest.importorskip("edge_tts")
    v = build_voice(_cfg(voice="edge", edge_voice="en-US-AnaNeural", edge_rate="+18%"))
    assert isinstance(v, voice.EdgeVoice) and (v.voice, v.rate) == ("en-US-AnaNeural", "+18%")
    assert isinstance(build_voice(_cfg(voice="silent")), SilentVoice)
    with pytest.raises(ValueError, match="Unknown voice provider"):
        build_voice(_cfg(voice="elevenlabs"))


# --- lineup ---------------------------------------------------------------------------------------------------------


def test_lineup_writes_one_mp3_per_working_voice_and_lists_the_settings(tmp_path, monkeypatch):
    made = _fake_edge(monkeypatch)
    out = tmp_path / "voices"
    ep = _episode()
    rows = lineup(_cfg(edge_rate="+18%", openai_api_key=""), ep, out,
                  voices=[("edge", "en-US-AnaNeural"), ("openai", "nova"), ("edge", "en-GB-MaisieNeural")])

    assert sorted(p.name for p in out.glob("*.mp3")) == ["01-edge-en-US-AnaNeural.mp3", "03-edge-en-GB-MaisieNeural.mp3"]
    assert [(v.voice, v.rate) for v in made] == [("en-US-AnaNeural", "+18%"), ("en-GB-MaisieNeural", "+18%")]
    assert all(v.spoken == [s.text for s in ep.segments] for v in made)

    expected_seconds = 3 * CLIP_SECONDS + 2 * voice.GAP
    ana, nova, maisie = rows
    assert ana["voice"] == "edge:en-US-AnaNeural" and ana["file"] == "01-edge-en-US-AnaNeural.mp3"
    assert ana["setting"] == "SHORTS_VOICE=edge and SHORTS_EDGE_VOICE=en-US-AnaNeural"
    assert ana["seconds"] == pytest.approx(expected_seconds, abs=0.2) and "error" not in ana
    assert media_duration(out / ana["file"]) == pytest.approx(expected_seconds, abs=0.2)
    assert maisie["file"] == "03-edge-en-GB-MaisieNeural.mp3"
    assert nova["voice"] == "openai:nova" and "file" not in nova and "seconds" not in nova
    assert nova["setting"] == "SHORTS_VOICE=openai and SHORTS_OPENAI_VOICE=nova"
    assert nova["error"].startswith("ValueError: ") and "OPENAI_API_KEY" in nova["error"]

    assert json.loads((out / "voices.json").read_text()) == rows
    text = (out / "voices.txt").read_text().splitlines()
    assert "01-edge-en-US-AnaNeural.mp3: 2 seconds" in text
    assert "    SHORTS_VOICE=edge and SHORTS_EDGE_VOICE=en-US-AnaNeural" in text
    assert f"openai:nova: {nova['error']}" in text
    assert "    SHORTS_VOICE=openai and SHORTS_OPENAI_VOICE=nova" in text
    assert "    SHORTS_VOICE=edge and SHORTS_EDGE_VOICE=en-GB-MaisieNeural" in text
    assert _work_dirs(out) == []


def test_lineup_records_a_voice_that_fell_back_to_silence_as_an_error(tmp_path, monkeypatch):
    _fake_edge(monkeypatch, fail_on="OpenAI shipped")
    out = tmp_path / "voices"
    [row] = lineup(_cfg(), _episode(), out, voices=[("edge", "en-US-JennyNeural")])
    assert "file" not in row
    assert row["error"] == "RuntimeError: the voice failed on 1 of 3 parts"
    assert list(out.glob("*.mp3")) == []
    assert "en-US-JennyNeural" in json.loads((out / "voices.json").read_text())[0]["voice"]
    assert "edge:en-US-JennyNeural: RuntimeError: the voice failed on 1 of 3 parts" in (out / "voices.txt").read_text()
    assert _work_dirs(out) == []


def test_lineup_never_raises_and_keeps_going_after_a_broken_voice(tmp_path, monkeypatch):
    made = []

    def factory(name, rate):
        made.append(FakeVoice("edge", name, garbage=name == "broken"))
        return made[-1]

    monkeypatch.setattr(voice, "EdgeVoice", factory)
    out = tmp_path / "voices"
    rows = lineup(_cfg(), _episode(), out, voices=[("edge", "broken"), ("piper", "amy"), ("edge", "en-US-AriaNeural")])
    broken, piper, aria = rows
    assert "error" in broken and "file" not in broken
    assert piper["error"] == "ValueError: unknown voice provider 'piper'"
    assert aria["file"] == "03-edge-en-US-AriaNeural.mp3"
    assert [p.name for p in out.glob("*.mp3")] == ["03-edge-en-US-AriaNeural.mp3"]
    assert _work_dirs(out) == []


def test_lineup_reads_the_configured_lineup_and_makes_safe_file_names(tmp_path, monkeypatch):
    _fake_edge(monkeypatch)
    out = tmp_path / "voices"
    rows = lineup(_cfg(voice_lineup=["edge:en US/Guy Neural", "edge:en US/Guy Neural", ""]), _episode(), out)
    assert [r["voice"] for r in rows] == ["edge:en US/Guy Neural"]
    assert rows[0]["file"] == "01-edge-en-US-Guy-Neural.mp3"
    assert (out / "01-edge-en-US-Guy-Neural.mp3").exists()


def test_lineup_openai_voice_uses_the_key_model_and_host_instructions(tmp_path, fake_openai):
    out = tmp_path / "voices"
    ep = _episode()
    [row] = lineup(_cfg(openai_api_key="sk-test", host_name="Quackers"), ep, out, voices=[("openai", "coral")])
    assert row["file"] == "01-openai-coral.mp3" and "error" not in row
    assert row["setting"] == "SHORTS_VOICE=openai and SHORTS_OPENAI_VOICE=coral"
    assert [c["input"] for c in fake_openai.calls] == [s.text for s in ep.segments]
    assert {(c["model"], c["voice"], c["response_format"]) for c in fake_openai.calls} == {
        ("gpt-4o-mini-tts", "coral", "mp3")}
    assert all(c["instructions"].startswith("You are Quackers, ") for c in fake_openai.calls)
    assert (out / "01-openai-coral.mp3").exists()


# --- CLI ------------------------------------------------------------------------------------------------------------


def test_cli_voices_reads_a_script_with_the_chosen_voice(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SHORTS_STATE_DIR", str(tmp_path / "state"))
    made = _fake_edge(monkeypatch)
    script = tmp_path / "03-episode.json"
    ep = _episode()
    script.write_text(episode_json(ep))
    out = tmp_path / "voices"
    assert cli.main(["voices", "--script", str(script), "--voices", "edge:x", "--out", str(out)]) == 0
    assert [p.name for p in out.glob("*.mp3")] == ["01-edge-x.mp3"]
    assert [r.get("file") for r in json.loads((out / "voices.json").read_text())] == ["01-edge-x.mp3"]
    assert made[0].spoken == [s.text for s in ep.segments]
    assert "1 of 1 voices done" in capsys.readouterr().out


def test_cli_voices_defaults_to_the_last_episode_and_every_voice(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("SHORTS_STATE_DIR", str(state))
    (state / "last_episode.json").write_text(episode_json(_episode()))
    seen = []
    monkeypatch.setattr(voice, "lineup", lambda cfg, episode, out_dir, voices: seen.append(
        (episode, out_dir, voices)) or [])
    out = tmp_path / "out"
    assert cli.main(["voices", "--out", str(out)]) == 0
    assert cli.main(["voices", "--out", str(out), "--voices", "edge:en-US-GuyNeural, nova"]) == 0
    (episode, out_dir, everyone), (_, _, two) = seen
    assert [s.text for s in episode.segments] == [s.text for s in _episode().segments]
    assert out_dir == out
    assert everyone == parse_lineup(["all"])
    assert two == [("edge", "en-US-GuyNeural"), ("openai", "nova")]


def test_cli_voices_without_a_script_returns_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SHORTS_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(voice, "lineup", lambda *a, **k: pytest.fail("lineup ran without a script"))
    missing = tmp_path / "nope.json"
    assert cli.main(["voices", "--script", str(missing), "--out", str(tmp_path / "v")]) == 1
    assert f"No script at {missing}" in capsys.readouterr().out
    assert cli.main(["voices", "--out", str(tmp_path / "v")]) == 1
    assert "last_episode.json" in capsys.readouterr().out
    assert not (tmp_path / "v").exists()


def test_python_m_shorts_voices_exits_1_without_a_script(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "SHORTS_STATE_DIR": str(tmp_path / "state"), "HOME": str(tmp_path)}
    proc = subprocess.run([sys.executable, "-m", "shorts", "voices", "--script", str(tmp_path / "missing.json"),
                           "--out", str(tmp_path / "v")], cwd=REPO, env=env, capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 1, proc.stderr
    assert "No script at" in proc.stdout
