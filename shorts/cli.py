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

    a = sub.add_parser("youtube-auth", help="one-time YouTube sign-in; prints the three YOUTUBE_* secrets")
    a.add_argument("client_secret", help="path to the OAuth client JSON from Google Cloud Console")

    v = sub.add_parser("voices", help="read a finished script with several voices, to compare them")
    v.add_argument("--script", help="a run's 03-episode.json (default: the last episode, state/last_episode.json)")
    v.add_argument("--voices", default="all",
                   help='"all", "edge", "openai", or a comma list like edge:en-US-AnaNeural,openai:coral')
    v.add_argument("--out", default="output/voices", help="where the MP3s go")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.cmd == "youtube-auth":
        from .upload import youtube_auth_flow

        secrets = youtube_auth_flow(args.client_secret)
        if not secrets["YOUTUBE_REFRESH_TOKEN"]:
            print("Google sent no refresh token. Remove the app's access at myaccount.google.com/permissions "
                  "and run this again.")
            return 1
        print("\nAdd these as repository secrets (GitHub: Settings, Secrets and variables, Actions, "
              "New repository secret), name on the left, value on the right of the = sign:\n")
        for name, value in secrets.items():
            print(f"{name}={value}")
        return 0

    if args.cmd == "voices":
        from pathlib import Path

        from .voice import lineup, parse_lineup
        from .writer import episode_from_json

        cfg = Config.from_env()
        script = Path(args.script) if args.script else cfg.state_dir / "last_episode.json"
        if not script.exists():
            print(f"No script at {script}; make an episode first or pass --script output/<run>/03-episode.json")
            return 1
        rows = lineup(cfg, episode_from_json(script.read_text()), Path(args.out),
                      parse_lineup(args.voices.split(",")))
        print(f"{sum('file' in r for r in rows)} of {len(rows)} voices done; see {Path(args.out) / 'voices.txt'}")
        return 0

    cfg = Config.from_env()
    if getattr(args, "offline", False):
        cfg = cfg.offline()
    if getattr(args, "stories", None):
        cfg.stories_per_video = args.stories

    from .notify import build_notifier, notify, run_url
    from .pipeline import run

    try:
        run(cfg, upload=getattr(args, "upload", False))
    except Exception as exc:
        url = run_url()
        where = (f"The run's log and download (anything it made, like short.mp4 if the video rendered): {url}\n\n"
                 if url else "")
        notify(build_notifier(cfg), f"{cfg.show_name} run failed: {exc}"[:150], where + traceback.format_exc())
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
