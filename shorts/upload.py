"""Stage 10: publish the video (or just record what would be published)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

from .checks import TITLE_MAX
from .config import Config
from .content import clean_url, is_banned_name
from .models import Episode, UploadResult

log = logging.getLogger(__name__)
YT_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
YT_SECRETS = ("YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET", "YOUTUBE_REFRESH_TOKEN")
# YouTube's limits: title 100 characters, description 5000 bytes, tags 500 characters in all; no < or > in any.
DESCRIPTION_MAX_BYTES = 4900
TAGS_MAX_CHARS = 450
_URL = re.compile(r"https?://\S+")
RETRY_STATUSES = (429, 500, 502, 503, 504)  # YouTube busy or down for a moment: resume the upload
UPLOAD_RETRIES = 5
RENEW_NOTICE_DAYS = 2  # the email starts reminding this many days before the sign-in ends
RENEW_STEPS = ("On your PC, open PowerShell in the Automated-AI-News folder, run "
               ".venv\\Scripts\\python -m shorts youtube-auth client_secret.json (on a Mac: "
               ".venv/bin/python -m shorts youtube-auth client_secret.json) and replace the "
               "YOUTUBE_REFRESH_TOKEN secret with the new value.")


def youtube_title(episode: Episode) -> str:
    """Title plus " #Shorts", cut so the hashtag always survives YouTube's 100 character limit."""
    title = " ".join(_plain(episode.title).split())[:TITLE_MAX].rstrip() or "AI News Today"
    return f"{title} #Shorts"


def _plain(text: str) -> str:
    """No < or >, which YouTube rejects in titles, descriptions and tags ("GPT-6 > GPT-5" becomes
    "GPT-6 \u203a GPT-5"). Line breaks stay."""
    lines = (re.sub(r"[ \t]+", " ", line.replace("<", "\u2039").replace(">", "\u203a")).strip()
             for line in (text or "").split("\n"))
    return "\n".join(lines).strip()


def youtube_description(episode: Episode) -> str:
    """The description with clean links (no newsletter tracking parameters), cut at a line to YouTube's size."""
    text = _URL.sub(lambda m: clean_url(m.group(0)), _plain(episode.description))
    if len(text.encode()) <= DESCRIPTION_MAX_BYTES:
        return text
    lines, out = text.split("\n"), ""
    for line in lines:
        candidate = f"{out}\n{line}" if out else line
        if len(candidate.encode()) > DESCRIPTION_MAX_BYTES:
            break
        out = candidate
    return out or text.encode()[:DESCRIPTION_MAX_BYTES].decode(errors="ignore")


def youtube_tags(episode: Episode) -> list[str]:
    """Tags YouTube accepts: no commas or < >, no newsletter or forum names, no repeats, 450 characters in all."""
    tags, seen, used = [], set(), 0
    for raw in episode.tags:
        for tag in _plain(str(raw)).split(","):
            tag = tag.strip().strip('"')
            if not tag or len(tag) > 100 or tag.lower() in seen or is_banned_name(tag):
                continue
            cost = len(tag) + (2 if " " in tag else 0) + (1 if tags else 0)  # quoted if it has a space, plus a comma
            if used + cost > TAGS_MAX_CHARS:
                continue
            tags.append(tag)
            seen.add(tag.lower())
            used += cost
    return tags


def upload_metadata(episode: Episode) -> dict:
    return {"title": youtube_title(episode), "description": youtube_description(episode),
            "tags": youtube_tags(episode)}


class Uploader(Protocol):
    name: str

    def upload(self, video: Path, episode: Episode) -> UploadResult: ...


class LocalUploader:
    """Default: leaves the MP4 in place and writes the metadata you'd paste into YouTube."""

    name = "local"

    def upload(self, video: Path, episode: Episode) -> UploadResult:
        return UploadResult(self.name, str(write_metadata(video, episode)))


def write_metadata(video: Path, episode: Episode) -> Path:
    """upload.json next to the video, and upload.txt with the same title, description and tags ready to
    paste into YouTube Studio for a hand upload. The YouTube uploader writes them too."""
    meta = upload_metadata(episode)
    path = video.with_name("upload.json")
    path.write_text(json.dumps({"file": video.name, **meta}, indent=2, ensure_ascii=False), encoding="utf-8")
    video.with_name("upload.txt").write_text(
        f"TITLE\n{meta['title']}\n\nDESCRIPTION\n{meta['description']}\n\nTAGS\n{', '.join(meta['tags'])}\n",
        encoding="utf-8")
    return path


class YouTubeSignInError(RuntimeError):
    """The YouTube secrets are missing, or Google refused them."""


class YouTubeUploader:
    """YouTube Data API v3 upload using an OAuth refresh token (see `shorts youtube-auth`)."""

    name = "youtube"

    def __init__(self, privacy: str):
        self.privacy = privacy
        missing = [k for k in YT_SECRETS if not os.environ.get(k, "").strip()]
        if missing:
            raise YouTubeSignInError(f"YouTube upload needs the secret {', '.join(missing)}")

    def credentials(self):
        from google.oauth2.credentials import Credentials

        return Credentials(
            None,
            refresh_token=os.environ["YOUTUBE_REFRESH_TOKEN"].strip(),
            client_id=os.environ["YOUTUBE_CLIENT_ID"].strip(),
            client_secret=os.environ["YOUTUBE_CLIENT_SECRET"].strip(),
            token_uri="https://oauth2.googleapis.com/token",
            scopes=YT_SCOPES,
        )

    def check(self) -> None:
        """Sign in before the episode is made, so an expired or revoked token is found in seconds, not
        after the render. Raises ``YouTubeSignInError`` with what to do about it. A network blip or a
        Google outage proves nothing, so it only logs and leaves the upload to try again at the end."""
        from google.auth.exceptions import RefreshError, TransportError
        from google.auth.transport.requests import Request

        try:
            self.credentials().refresh(Request())
        except RefreshError as exc:
            if getattr(exc, "retryable", False):
                log.warning("the YouTube sign-in check didn't get an answer (%s); trying again at upload", exc)
                return
            raise YouTubeSignInError(f"Google refused the saved YouTube sign-in ({exc})") from exc
        except TransportError as exc:
            log.warning("the YouTube sign-in check couldn't reach Google (%s); trying again at upload", exc)

    def upload(self, video: Path, episode: Episode) -> UploadResult:
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
        from googleapiclient.http import MediaFileUpload

        write_metadata(video, episode)
        meta = upload_metadata(episode)
        yt = build("youtube", "v3", credentials=self.credentials(), cache_discovery=False)
        req = yt.videos().insert(
            part="snippet,status",
            body={
                "snippet": {
                    **meta,
                    "categoryId": "28",  # Science & Technology
                    "defaultLanguage": "en",
                    "defaultAudioLanguage": "en",
                },
                "status": {"privacyStatus": self.privacy, "selfDeclaredMadeForKids": False,
                           "containsSyntheticMedia": True},
            },
            media_body=MediaFileUpload(str(video), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True),
        )
        resp, drops = None, 0
        while resp is None:
            try:
                # No retries inside the client: it would resend a chunk it has already read (an empty body).
                # After an error the next call asks YouTube how much arrived and resumes from there.
                _, resp = req.next_chunk(num_retries=0)
            except (OSError, HttpError, _http_errors()) as exc:
                status = getattr(getattr(exc, "resp", None), "status", None)
                if isinstance(exc, HttpError) and status not in RETRY_STATUSES:
                    raise  # a rejection (bad metadata, quota): trying again won't help
                drops += 1
                if drops > UPLOAD_RETRIES:
                    raise
                log.warning("      upload interrupted (%s); resuming", exc)
                time.sleep(5 * drops)
        return UploadResult(self.name, f"https://youtube.com/shorts/{resp['id']}")


def _http_errors() -> type[Exception]:
    try:
        from httplib2 import HttpLib2Error

        return HttpLib2Error
    except ImportError:  # pragma: no cover - httplib2 comes with google-api-python-client
        return OSError


def build_uploader(cfg: Config) -> Uploader:
    if cfg.uploader == "youtube":
        return YouTubeUploader(cfg.youtube_privacy)
    if cfg.uploader != "local":
        raise ValueError(f"Unknown uploader {cfg.uploader!r}")
    return LocalUploader()


def upload_hint(exc: Exception) -> str:
    """What went wrong with a YouTube upload, and what to do about it, for the email."""
    text = str(exc)
    if isinstance(exc, YouTubeSignInError) and "needs the secret" in text:
        return (f"{text}. Add it under Settings, Secrets and variables, Actions in the GitHub repository "
                "(see the README's YouTube upload setup).")
    if isinstance(exc, YouTubeSignInError) or "invalid_grant" in text:
        return (f"{text[:300]}. The YouTube sign-in expired or was revoked. {RENEW_STEPS} While the Google "
                "Cloud app is in Testing, Google ends each sign-in 7 days after you sign in, so this is "
                "expected once a week.")
    if "quotaExceeded" in text or "uploadLimitExceeded" in text:
        return f"YouTube's daily upload limit was reached ({text[:200]}). Tomorrow's run uploads as usual."
    return f"{type(exc).__name__}: {text[:300]}"


class SignInLog:
    """When this run first saw the current YouTube sign-in, so the email can say when to renew it.

    Google ends sign-ins 7 days after consent while the Google Cloud app is in Testing, and nothing in
    the token says when. The state branch is public, so it keeps a short fingerprint of the refresh
    token, never the token itself.
    """

    def __init__(self, path: Path):
        self.path = path

    def since(self, token: str, today: date) -> date:
        """The day this token was first seen, recording today if it is new."""
        mark = hashlib.sha256(token.encode()).hexdigest()[:12]
        try:
            data = json.loads(self.path.read_text())
            if data.get("token") == mark:
                return date.fromisoformat(data["since"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"token": mark, "since": today.isoformat()}) + "\n")
        return today


def renew_reminder(cfg: Config, today: date | None = None) -> str:
    """"" or a line for the email when the YouTube sign-in ends within RENEW_NOTICE_DAYS days.

    The sign-in may have been made the day before this run first saw it, so the last safe day is
    a day early: a sign-in first seen on Monday is renewed by Sunday.
    """
    token = os.environ.get("YOUTUBE_REFRESH_TOKEN", "").strip()
    if cfg.youtube_signin_days <= 0 or not token:
        return ""
    today = today or datetime.now(timezone.utc).date()
    since = SignInLog(cfg.state_dir / "youtube_signin.json").since(token, today)
    last = since + timedelta(days=cfg.youtube_signin_days - 1)
    left = (last - today).days
    if not 0 <= left <= RENEW_NOTICE_DAYS:  # still signed in after the end: the app was published
        return ""
    when = ("today, or tomorrow's upload may fail" if left == 0 else "by tomorrow" if left == 1
            else f"by {last:%A, %B} {last.day}")
    return (f"Renew the YouTube sign-in {when}. Google ends it {cfg.youtube_signin_days} days after you sign in "
            f"while the Google Cloud app is in Testing. {RENEW_STEPS}")


def _client_ids(client_secret_file: str) -> tuple[str, str]:
    data = json.loads(Path(client_secret_file).read_text())
    app = data.get("installed") or data.get("web") or {}
    return app.get("client_id", ""), app.get("client_secret", "")


def youtube_auth_flow(client_secret_file: str) -> dict[str, str]:
    """One-time interactive login on your own machine: the three values for the GitHub secrets."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, YT_SCOPES)
    creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
    client_id, client_secret = _client_ids(client_secret_file)
    return {"YOUTUBE_CLIENT_ID": client_id, "YOUTUBE_CLIENT_SECRET": client_secret,
            "YOUTUBE_REFRESH_TOKEN": creds.refresh_token or ""}
