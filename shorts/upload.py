"""Stage 10: publish the video (or just record what would be published)."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Protocol

from .config import Config
from .models import Episode, UploadResult

log = logging.getLogger(__name__)
YT_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


class Uploader(Protocol):
    name: str

    def upload(self, video: Path, episode: Episode) -> UploadResult: ...


class LocalUploader:
    """Default: leaves the MP4 in place and writes the metadata you'd paste into YouTube."""

    name = "local"

    def upload(self, video: Path, episode: Episode) -> UploadResult:
        meta = video.with_name("upload.json")
        meta.write_text(json.dumps({
            "file": video.name,
            "title": f"{episode.title} #Shorts",
            "description": episode.description,
            "tags": episode.tags,
        }, indent=2))
        return UploadResult(self.name, str(meta))


class YouTubeUploader:
    """YouTube Data API v3 upload using an OAuth refresh token (see `shorts youtube-auth`)."""

    name = "youtube"

    def __init__(self, privacy: str):
        self.privacy = privacy
        missing = [k for k in ("YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET", "YOUTUBE_REFRESH_TOKEN")
                   if not os.environ.get(k)]
        if missing:
            raise RuntimeError(f"YouTube upload needs {', '.join(missing)}")

    def upload(self, video: Path, episode: Episode) -> UploadResult:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

        creds = Credentials(
            None,
            refresh_token=os.environ["YOUTUBE_REFRESH_TOKEN"],
            client_id=os.environ["YOUTUBE_CLIENT_ID"],
            client_secret=os.environ["YOUTUBE_CLIENT_SECRET"],
            token_uri="https://oauth2.googleapis.com/token",
            scopes=YT_SCOPES,
        )
        yt = build("youtube", "v3", credentials=creds)
        req = yt.videos().insert(
            part="snippet,status",
            body={
                "snippet": {
                    "title": f"{episode.title} #Shorts"[:100],
                    "description": episode.description[:4900],
                    "tags": episode.tags,
                    "categoryId": "28",  # Science & Technology
                },
                "status": {"privacyStatus": self.privacy, "selfDeclaredMadeForKids": False,
                           "containsSyntheticMedia": True},
            },
            media_body=MediaFileUpload(str(video), mimetype="video/mp4", resumable=True),
        )
        resp = None
        while resp is None:
            _, resp = req.next_chunk()
        return UploadResult(self.name, f"https://youtube.com/shorts/{resp['id']}")


def build_uploader(cfg: Config) -> Uploader:
    if cfg.uploader == "youtube":
        return YouTubeUploader(cfg.youtube_privacy)
    if cfg.uploader != "local":
        raise ValueError(f"Unknown uploader {cfg.uploader!r}")
    return LocalUploader()


def youtube_auth_flow(client_secret_file: str) -> str:
    """One-time interactive login on your own machine; prints the refresh token to put in .env."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, YT_SCOPES)
    creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
    return creds.refresh_token
