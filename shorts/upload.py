"""Stage 10: publish the video (or just record what would be published)."""
from __future__ import annotations

import json
import logging
import os
import re
import time
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
    """upload.json next to the video: what a hand upload needs (the YouTube uploader writes it too)."""
    meta = video.with_name("upload.json")
    meta.write_text(json.dumps({"file": video.name, **upload_metadata(episode)}, indent=2))
    return meta


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
        after the render. Raises ``YouTubeSignInError`` with what to do about it."""
        from google.auth.exceptions import RefreshError
        from google.auth.transport.requests import Request

        try:
            self.credentials().refresh(Request())
        except RefreshError as exc:
            raise YouTubeSignInError(f"Google refused the saved YouTube sign-in ({exc})") from exc

    def upload(self, video: Path, episode: Episode) -> UploadResult:
        from googleapiclient.discovery import build
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
                # Retries 5xx and rate limits itself and resumes the same upload session.
                _, resp = req.next_chunk(num_retries=5)
            except (OSError, _http_errors()) as exc:  # a dropped connection mid-chunk: resume where it stopped
                drops += 1
                if drops > 3:
                    raise
                log.warning("      upload connection dropped (%s); resuming", exc)
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
        return (f"{text[:300]}. The YouTube sign-in expired or was revoked. On your PC, run "
                "python -m shorts youtube-auth client_secret.json again and replace the YOUTUBE_REFRESH_TOKEN "
                "secret with the new value. If the Google Cloud app is still in Testing, publish it "
                "(In production) first, or the new sign-in also stops working after 7 days.")
    if "quotaExceeded" in text or "uploadLimitExceeded" in text:
        return f"YouTube's daily upload limit was reached ({text[:200]}). Tomorrow's run uploads as usual."
    return f"{type(exc).__name__}: {text[:300]}"


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
