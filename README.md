# Duck Desk: automated AI news Shorts

Every day this pipeline reads the AI newsletters that arrived in its inbox plus AI news
feeds, picks the 8 biggest stories, writes a ~2 minute script, has a cartoon duck anchor
read it, and uploads the vertical video to YouTube.

```
gather news -> pick 8 -> read articles -> write episode -> voice -> animate duck -> cards -> render -> QA -> upload -> email
sources.py    selection   research.py     writer.py        voice.py  character.py    visuals   composer   qa.py  upload.py  notify.py
```

Every stage is a small interface with a factory that picks the implementation from
config, and every stage that needs a key or the network falls back to a free or offline
default, so a run always finishes when there is real news to report.

With an LLM key, the two judgement stages run as agents (`SHORTS_AGENTS=on`, the default):

- **Editor agent** (`selection.py`) can search the day's candidates and the list of stories
  that already aired while it picks. Its picks are then checked in code (`checks.py`): no
  duplicates, no repeats, only links that appear in the sources, nothing too old. Problems go
  back to it for up to `SHORTS_MAX_REPAIRS` rounds; picks that still fail are dropped and
  replaced from the keyword-scored pool.
- **Writer agent** (`writer.py`) drafts the script, then a critic model checks every claim
  against the story material while code checks length, hype words, links in the narration
  and numbers that aren't in the sources. Only the failing segments are rewritten; any that
  still fail get the template line for that story, not a whole-script fallback.

Every LLM call is priced and logged in `cost.json`. A run stops calling the LLM at
`SHORTS_BUDGET_USD` (default $0.60) and the month at `SHORTS_MONTHLY_BUDGET_USD` (default $18,
kept in `state/spend.json`); stages then use their no-key fallbacks.

A run fails instead of uploading when every news source is down, when there aren't enough
fresh stories, when the voice left a segment silent, or when a sample story slipped in.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
cp .env.example .env         # fill in what you have; everything is optional

python -m shorts run --offline   # no network, no keys: sample news, silent voice
python -m shorts run             # live news, writes output/<timestamp>/
python -m shorts run --upload    # also publishes, if SHORTS_UPLOADER=youtube
pytest
```

Each run writes `output/<timestamp>/`:

| File | What it is |
|------|------------|
| `01-candidates.json` | everything gathered: newsletter issues, feed and forum headlines (`.full.json` keeps the text) |
| `02-picks.json` | the 8 stories the editor chose, with outlets, key fact and article text |
| `03-episode.json` | the script: intro, one segment per story, outro, title, description |
| `audio/` | one voice clip per segment and the joined `voice.wav` |
| `host/`, `cards/` | duck sprites and per-segment backgrounds |
| `timeline.json`, `captions.ass` | what the renderer draws, and when |
| `short.mp4` | the 1080x1920 video |
| `qa.json`, `cost.json`, `upload.json` | checks, LLM calls and dollars spent, YouTube metadata |

## Stages and how to swap them

| Stage | Default | Needs | Fallback | Upgrade path |
|-------|---------|-------|----------|--------------|
| News | Newsletter inbox (AgentMail), RSS, Hacker News, Reddit (RSS) | `AGENTMAIL_API_KEY`, `AGENTMAIL_INBOX` | feeds only; the run fails if nothing arrives | more feeds, X lists |
| Pick | editor agent with search tools and checked picks (`SHORTS_EDITOR_MODEL`) | `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`) | keyword + freshness scoring | web search, analytics feedback |
| Read | Tavily extracts the full article for each pick | `TAVILY_API_KEY` | newsletter summary only | |
| Script | writer agent + critic (`SHORTS_WRITER_MODEL`, `SHORTS_CHECKER_MODEL`), persona in `shorts/persona.md` | LLM key | template line per failing segment | LangGraph newsroom with approval |
| Voice | edge-tts `en-US-AnaNeural` (free); script trimmed if the audio runs past 170 s | internet | silence, which fails QA | ElevenLabs designed voice |
| Host | puppet duck, bill moves with the voice loudness | – | – | Kling AI Avatar (`SHORTS_ANIMATOR`) |
| Render | cards + duck + desk + karaoke captions, one ffmpeg call | – | – | Remotion |
| Upload | `local` (metadata only) or `youtube` (private, marked synthetic) | YouTube OAuth secrets | – | public after API audit |
| Notify | email from the AgentMail inbox | `SHORTS_NOTIFY_EMAIL` | log line | Telegram approve button |

To add an implementation, write a class with the stage's method and register it in that
module's `build_*` function.

## Running daily on GitHub Actions

`.github/workflows/daily-short.yml` runs at 12:47 UTC and can be started by hand from the
Actions tab. It keeps the list of already-aired stories and the month's LLM spend on a `state`
branch and saves the video and JSON files as a workflow artifact.

In the repository settings, under **Secrets and variables → Actions**, add:

| Name | Kind | Value |
|------|------|-------|
| `OPENAI_API_KEY` | secret | OpenAI key |
| `AGENTMAIL_API_KEY` | secret | AgentMail key |
| `TAVILY_API_KEY` | secret | Tavily key |
| `AGENTMAIL_INBOX` | variable | `agentnews247@agentmail.to` |
| `SHORTS_NOTIFY_EMAIL` | variable | where the "episode ready" email goes |
| `SHORTS_LLM_MODEL` | variable | optional; one model for every role (default: editor and writer `gpt-6-sol`, checker `gpt-6-luna`, `gpt-5` if those aren't available) |
| `SHORTS_EDITOR_MODEL`, `SHORTS_WRITER_MODEL`, `SHORTS_CHECKER_MODEL` | variables | optional, one role's model |
| `SHORTS_AGENTS`, `SHORTS_MAX_REPAIRS` | variables | optional, `off` for one-shot LLM calls; repair rounds per agent (default `2`) |
| `SHORTS_BUDGET_USD`, `SHORTS_MONTHLY_BUDGET_USD` | variables | optional, default `0.60` and `18` |
| `SHORTS_UPLOADER` | variable | `youtube` once the YouTube secrets are in |
| `YOUTUBE_CLIENT_ID`, `YOUTUBE_CLIENT_SECRET`, `YOUTUBE_REFRESH_TOKEN` | secrets | see below |

## YouTube upload setup (one time)

1. In Google Cloud Console, create a project and enable **YouTube Data API v3**.
2. Configure the OAuth consent screen (External, scope `youtube.upload`) and set its
   publishing status to **In production**. In *Testing* mode refresh tokens expire after
   7 days and the daily upload stops working.
3. Create an **OAuth client ID** of type *Desktop app* and download the JSON.
4. On your own machine: `python -m shorts youtube-auth client_secret.json`, sign in with
   the channel's Google account, and store the printed `YOUTUBE_REFRESH_TOKEN` plus the
   client ID and secret as repository secrets.
5. Set the variable `SHORTS_UPLOADER=youtube`. Videos upload as `private`: until the
   Google Cloud project passes YouTube's API audit, API uploads can only be private.
