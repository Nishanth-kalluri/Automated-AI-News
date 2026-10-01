"""The weekly YouTube sign-in while the Google Cloud app is in Testing: when to renew, and the email."""
import hashlib
import json
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

from shorts import pipeline
from shorts.config import Config
from shorts.upload import SignInLog, YouTubeSignInError, renew_reminder, upload_hint
from tests.test_shadow import _stub_run

MON = date(2026, 10, 5)


def _cfg(tmp_path, days=7):
    return replace(Config.from_env(), state_dir=tmp_path / "state", youtube_signin_days=days)


def test_the_log_keeps_the_first_day_a_sign_in_was_seen(tmp_path):
    log = SignInLog(tmp_path / "state" / "youtube_signin.json")
    assert log.since("token-1", MON) == MON
    assert log.since("token-1", date(2026, 10, 9)) == MON
    assert log.since("token-2", date(2026, 10, 9)) == date(2026, 10, 9)  # a new sign-in starts over


def test_the_public_state_never_holds_the_token(tmp_path):
    path = tmp_path / "youtube_signin.json"
    SignInLog(path).since("1//0secret-refresh-token", MON)
    data = json.loads(path.read_text())
    assert "secret" not in path.read_text() and len(data["token"]) == 12 and data["since"] == "2026-10-05"


def test_a_broken_log_starts_over(tmp_path):
    path = tmp_path / "youtube_signin.json"
    mark = hashlib.sha256(b"token-1").hexdigest()[:12]
    for junk in ("", "[]", "{not json", json.dumps({"token": mark}), json.dumps({"token": mark, "since": "soon"})):
        path.write_text(junk)
        assert SignInLog(path).since("token-1", MON) == MON


def test_the_reminder_starts_two_days_before_the_sign_in_ends(monkeypatch, tmp_path):
    monkeypatch.setenv("YOUTUBE_REFRESH_TOKEN", " token-1 ")
    cfg = _cfg(tmp_path)
    assert renew_reminder(cfg, MON) == ""  # first seen Monday: renew by Sunday, a day early to be safe
    assert renew_reminder(cfg, date(2026, 10, 8)) == ""  # Thursday, 3 days left
    friday = renew_reminder(cfg, date(2026, 10, 9))
    assert friday.startswith("Renew the YouTube sign-in by Sunday, October 11.")
    assert "youtube-auth" in friday and "YOUTUBE_REFRESH_TOKEN" in friday and "7 days" in friday
    assert renew_reminder(cfg, date(2026, 10, 10)).startswith("Renew the YouTube sign-in by tomorrow.")
    assert "today, or tomorrow's upload may fail" in renew_reminder(cfg, date(2026, 10, 11))
    assert renew_reminder(cfg, date(2026, 10, 12)) == ""  # still signed in after the end: the app was published
    monkeypatch.setenv("YOUTUBE_REFRESH_TOKEN", "token-2")  # renewed on Saturday
    assert renew_reminder(cfg, date(2026, 10, 10)) == ""


def test_no_reminder_for_a_published_app_or_without_a_sign_in(monkeypatch, tmp_path):
    monkeypatch.setenv("YOUTUBE_REFRESH_TOKEN", "token-1")
    assert renew_reminder(_cfg(tmp_path, days=0), MON) == ""
    assert not (tmp_path / "state" / "youtube_signin.json").exists()
    monkeypatch.setenv("YOUTUBE_REFRESH_TOKEN", " ")
    assert renew_reminder(_cfg(tmp_path), MON) == ""


def test_the_sign_in_length_is_a_setting(monkeypatch):
    assert Config.from_env().youtube_signin_days == 7
    monkeypatch.setenv("SHORTS_YOUTUBE_SIGNIN_DAYS", "0")
    assert Config.from_env().youtube_signin_days == 0


def test_an_expired_sign_in_says_it_is_the_weekly_renewal():
    hint = upload_hint(YouTubeSignInError("Google refused the saved YouTube sign-in (invalid_grant)"))
    assert "youtube-auth" in hint and "YOUTUBE_REFRESH_TOKEN" in hint and "once a week" in hint
    assert "publish" not in hint.lower()


def test_the_episode_email_reminds_before_the_sign_in_ends(monkeypatch, tmp_path):
    rig = _stub_run(monkeypatch, tmp_path)
    monkeypatch.setenv("YOUTUBE_REFRESH_TOKEN", "token-1")
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).date()
    SignInLog(state / "youtube_signin.json").since("token-1", today - timedelta(days=6))
    pipeline.run(replace(rig.cfg, shadow=False), upload=True)
    subject, text = rig.notes[-1]
    assert subject.startswith("New episode ready, renew the YouTube sign-in: ")
    assert text.startswith("Uploaded to YouTube as private:")
    assert "Renew the YouTube sign-in today" in text and "youtube-auth" in text


def test_a_fresh_sign_in_is_recorded_without_a_reminder(monkeypatch, tmp_path):
    rig = _stub_run(monkeypatch, tmp_path)
    monkeypatch.setenv("YOUTUBE_REFRESH_TOKEN", "token-1")
    pipeline.run(replace(rig.cfg, shadow=False), upload=True)
    subject, text = rig.notes[-1]
    assert subject.startswith("New episode ready: ") and "Renew" not in text
    assert json.loads((tmp_path / "state" / "youtube_signin.json").read_text())["token"]


def test_no_reminder_when_the_run_did_not_upload(monkeypatch, tmp_path):
    rig = _stub_run(monkeypatch, tmp_path)
    monkeypatch.setenv("YOUTUBE_REFRESH_TOKEN", "token-1")
    pipeline.run(replace(rig.cfg, shadow=False), upload=False)
    assert "Renew" not in rig.notes[-1][1]
    assert not (tmp_path / "state" / "youtube_signin.json").exists()
