from __future__ import annotations

import argparse
import logging
import sys
import traceback

from .config import Config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="shorts", description="AI news -> YouTube Short")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="run the full pipeline (default)")
    r.add_argument("--upload", action="store_true",
                   help="publish with SHORTS_UPLOADER (otherwise only writes upload.json)")
    r.add_argument("--offline", action="store_true",
                   help="sample news, template script, silent voice: no network or keys needed")
    r.add_argument("--stories", type=int, help="stories per video")

    a = sub.add_parser("youtube-auth", help="one-time OAuth login; prints YOUTUBE_REFRESH_TOKEN")
    a.add_argument("client_secret", help="path to the OAuth client JSON from Google Cloud Console")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.cmd == "youtube-auth":
        from .upload import youtube_auth_flow

        print("YOUTUBE_REFRESH_TOKEN=" + youtube_auth_flow(args.client_secret))
        return 0

    cfg = Config.from_env()
    if getattr(args, "offline", False):
        cfg = cfg.offline()
    if getattr(args, "stories", None):
        cfg.stories_per_video = args.stories

    from .notify import build_notifier, notify
    from .pipeline import run

    try:
        run(cfg, upload=getattr(args, "upload", False))
    except Exception as exc:
        notify(build_notifier(cfg), f"{cfg.show_name} run failed: {exc}"[:150], traceback.format_exc())
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
