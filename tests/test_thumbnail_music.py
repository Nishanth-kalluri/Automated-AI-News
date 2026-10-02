"""The episode thumbnail and how it reaches YouTube, and the background music under the voice."""
import json
import re
import subprocess
import sys
import types
from dataclasses import replace
from datetime import date, timedelta

import pytest

from shorts import composer, pipeline, upload
from shorts.config import Config
from shorts.media import ffmpeg_exe, has_audio
from shorts.models import Episode, Segment, UploadResult
from shorts.music import pick_track, tracks
from shorts.thumbnail import draw_thumbnail, save_thumbnail
from shorts.upload import YouTubeUploader, thumbnail_hint, write_metadata
from shorts.visuals import ORANGE
from tests.test_intro_outro_youtube import _episode, _http_error, _Request, _secrets
from tests.test_shadow import _stub_run

DAY = date(2026, 10, 2)


def _show(*headlines):
    stories = [Segment("story", "Text.", h) for h in headlines]
    return Episode(title="t", description="d", tags=[], segments=[Segment("intro", "Hi."), *stories,
                                                                   Segment("outro", "Bye.")])


# --- the thumbnail -----------------------------------------------------------------------------

def test_the_thumbnail_is_a_vertical_png_youtube_accepts(tmp_path):
    path = save_thumbnail(_show("OpenAI ships an agent that plans whole trips", "A faster chip", "A new lab"),
                          "Duck Desk", tmp_path / "thumbnail.png", DAY)
    img = draw_thumbnail(_show("x"), "Duck Desk", DAY)
    assert img.size == (1080, 1920) and img.getpixel((80, 285)) == ORANGE  # the "TODAY IN AI" pill
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and path.stat().st_size < 2_000_000  # YouTube's limit


def test_any_headline_fits_and_an_episode_without_stories_still_draws():
    long = "An Extremely Long Headline About A Model That Goes On And On " * 4
    assert draw_thumbnail(_show(long, "Two"), "Duck Desk", DAY).size == (1080, 1920)
    assert draw_thumbnail(_show(), "Duck Desk", DAY).size == (1080, 1920)


def test_a_hand_upload_is_told_about_the_thumbnail(tmp_path):
    video = tmp_path / "short.mp4"
    video.write_bytes(b"x")
    write_metadata(video, _episode())
    assert "THUMBNAIL" not in (tmp_path / "upload.txt").read_text()
    (tmp_path / "thumbnail.png").write_bytes(b"png")
    write_metadata(video, _episode())
    assert (tmp_path / "upload.txt").read_text().endswith("\nTHUMBNAIL\nthumbnail.png, in the same folder\n")


def _fake_google(monkeypatch, request, thumbnail_error=None):
    calls = {"thumbnails": []}

    class Videos:
        def insert(self, **kw):
            return request

    class Set:
        def __init__(self, kw):
            self.kw = kw

        def execute(self, num_retries=0):
            calls["thumbnails"].append(self.kw)
            if thumbnail_error:
                raise thumbnail_error
            return {}

    class Thumbnails:
        def set(self, **kw):
            return Set(kw)

    class YouTube:
        def videos(self):
            return Videos()

        def thumbnails(self):
            return Thumbnails()

    discovery = types.ModuleType("googleapiclient.discovery")
    discovery.build = lambda *a, **kw: YouTube()
    http = types.ModuleType("googleapiclient.http")
    http.MediaFileUpload = lambda path, **kw: (path, kw)
    monkeypatch.setitem(sys.modules, "googleapiclient.discovery", discovery)
    monkeypatch.setitem(sys.modules, "googleapiclient.http", http)
    monkeypatch.setattr(upload.time, "sleep", lambda s: None)
    return calls


def _video(tmp_path, thumbnail=True):
    video = tmp_path / "short.mp4"
    video.write_bytes(b"x")
    if thumbnail:
        (tmp_path / "thumbnail.png").write_bytes(b"png")
    return video


def test_the_upload_sets_the_thumbnail(monkeypatch, tmp_path):
    _secrets(monkeypatch)
    calls = _fake_google(monkeypatch, _Request([{"id": "abc123"}]))
    result = YouTubeUploader("private").upload(_video(tmp_path), _episode())
    assert result == UploadResult("youtube", "https://youtube.com/shorts/abc123")
    [sent] = calls["thumbnails"]
    assert sent["videoId"] == "abc123"
    assert sent["media_body"] == (str(tmp_path / "thumbnail.png"), {"mimetype": "image/png"})


def test_without_a_thumbnail_file_nothing_is_set(monkeypatch, tmp_path):
    _secrets(monkeypatch)
    calls = _fake_google(monkeypatch, _Request([{"id": "abc123"}]))
    assert YouTubeUploader("private").upload(_video(tmp_path, thumbnail=False), _episode()).note == ""
    assert calls["thumbnails"] == []


def test_a_refused_thumbnail_keeps_the_upload_and_says_what_to_do(monkeypatch, tmp_path):
    _secrets(monkeypatch)
    _fake_google(monkeypatch, _Request([{"id": "abc123"}]), thumbnail_error=_http_error(403))
    result = YouTubeUploader("private").upload(_video(tmp_path), _episode())
    assert result.location == "https://youtube.com/shorts/abc123"
    assert "https://www.youtube.com/verify" in result.note and "thumbnail.png" in result.note
    other = thumbnail_hint(ConnectionResetError("reset"))
    assert "ConnectionResetError: reset" in other and "verify" not in other and "thumbnail.png" in other


def test_a_run_draws_the_thumbnail_and_the_email_passes_on_the_uploaders_note(monkeypatch, tmp_path):
    rig = _stub_run(monkeypatch, tmp_path)
    note = "YouTube didn't take the thumbnail: verify the channel."

    class NotedYouTube:
        name = "youtube"

        def upload(self, video, episode):
            return UploadResult("youtube", "https://youtube.com/shorts/abc", note)

    monkeypatch.setattr(pipeline, "build_uploader", lambda cfg: NotedYouTube())
    video = pipeline.run(replace(rig.cfg, shadow=False), upload=True)
    assert (video.parent / "thumbnail.png").stat().st_size > 10_000
    assert note in rig.notes[-1][1]


def test_the_workflow_keeps_the_thumbnail_in_the_download():
    from pathlib import Path

    flow = (Path(__file__).parent.parent / ".github" / "workflows" / "daily-short.yml").read_text()
    assert "output/*/thumbnail.png" in flow


# --- the music ---------------------------------------------------------------------------------

def test_each_day_takes_the_next_track_in_name_order(tmp_path):
    for name in ("b-sunny.mp3", "a-bounce.m4a", "c-pond.WAV", "README.md", "cover.png"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "old.mp3").mkdir()  # a folder, not a track
    assert [p.name for p in tracks(tmp_path)] == ["a-bounce.m4a", "b-sunny.mp3", "c-pond.WAV"]
    week = [pick_track(tmp_path, DAY + timedelta(days=d)).name for d in range(6)]
    assert week[:3] == week[3:] and sorted(week[:3]) == ["a-bounce.m4a", "b-sunny.mp3", "c-pond.WAV"]


def test_no_tracks_means_no_music(tmp_path):
    assert pick_track(tmp_path / "missing", DAY) is None
    assert pick_track(tmp_path, DAY) is None
    assert pick_track(None, DAY) is None


def test_music_settings(monkeypatch):
    cfg = Config.from_env()
    assert str(cfg.music_dir) == "assets/music" and cfg.music_volume == 0.15
    monkeypatch.setenv("SHORTS_MUSIC_DIR", "/music")
    monkeypatch.setenv("SHORTS_MUSIC_VOLUME", "0.1")
    assert str(Config.from_env().music_dir) == "/music" and Config.from_env().music_volume == 0.1
    monkeypatch.setenv("SHORTS_MUSIC", "off")
    assert Config.from_env().music_dir is None
    monkeypatch.setenv("SHORTS_MUSIC_VOLUME", "15")
    with pytest.raises(ValueError, match="between 0 and 1"):
        Config.from_env()


def test_the_music_fades_and_ducks_under_the_voice():
    graph = composer.audio_graph(120.0, 0.15)
    assert "volume=0.150" in graph and "afade=t=in:d=1.0" in graph and "afade=t=out:st=117.500:d=2.5" in graph
    assert "[bed][key]sidechaincompress" in graph  # the voice pushes the music down
    assert "amix=inputs=2:duration=first:normalize=0" in graph and graph.endswith(f"{composer.LOUDNESS}[a]")


def test_a_run_mixes_todays_track_and_the_email_names_it(monkeypatch, tmp_path):
    rig = _stub_run(monkeypatch, tmp_path)
    music = tmp_path / "music"
    music.mkdir()
    for name in ("a-bounce.mp3", "b-sunny.mp3"):
        (music / name).write_bytes(b"x")
    pipeline.run(replace(rig.cfg, shadow=False, music_dir=music), upload=False)
    today = pick_track(music)
    assert rig.music == [today] and f"music: {today.stem}" in rig.notes[-1][1]
    pipeline.run(replace(rig.cfg, shadow=False, music_dir=None, state_dir=tmp_path / "fresh"), upload=False)
    assert rig.music[-1] is None and "music:" not in rig.notes[-1][1]


def test_a_track_ffmpeg_cannot_play_costs_the_music_not_the_episode(monkeypatch, tmp_path):
    rig = _stub_run(monkeypatch, tmp_path)
    music = tmp_path / "music"
    music.mkdir()
    (music / "broken.mp3").write_bytes(b"not audio")
    calls = []

    def render(vo, cards, desk, host, out, preset, music=None, music_volume=0.0):
        calls.append(music)
        if music:
            raise RuntimeError("ffmpeg failed: Invalid data found when processing input")
        return out

    monkeypatch.setattr(pipeline.composer, "render", render)
    pipeline.run(replace(rig.cfg, shadow=False, music_dir=music), upload=False)
    assert calls == [music / "broken.mp3", None]
    assert "no music (broken.mp3 couldn't be played)" in rig.notes[-1][1]


def _mean_volume(path) -> float:
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
                          capture_output=True, text=True)
    return float(re.search(r"mean_volume: (-?[\d.]+) dB", proc.stderr).group(1))


def test_a_real_render_mixes_the_music_under_the_voice(tmp_path, monkeypatch):
    music = tmp_path / "music"
    music.mkdir()
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=4", str(music / "tone.wav")], check=True)
    monkeypatch.setenv("SHORTS_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("SHORTS_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("SHORTS_X264_PRESET", "ultrafast")
    monkeypatch.setenv("SHORTS_MUSIC_DIR", str(music))
    video = pipeline.run(Config.from_env().offline())  # a silent voice, so only the music is heard
    assert has_audio(video) and _mean_volume(video) > -40  # the 4 s tone loops for the whole episode
    timeline = json.loads((video.parent / "timeline.json").read_text())
    assert timeline["music"] == {"track": "tone.wav", "volume": 0.15}
    assert (video.parent / "thumbnail.png").exists()
