# Duck Desk: automated AI news Shorts

Every day this pipeline reads the AI newsletters that arrived in its inbox plus AI news
feeds, picks up to 8 of the biggest stories, writes a ~2 minute script, has a cartoon duck
anchor read it, and uploads the vertical video to YouTube.

```
gather news -> pick up to 8 -> read articles -> write episode -> voice -> animate duck -> cards -> render -> QA -> upload -> email
sources.py    selection   research.py     writer.py        voice.py  character.py    visuals   composer   qa.py  upload.py  notify.py
```

Every stage is a small interface with a factory that picks the implementation from
config, and most stages that need a key or the network fall back to a free or offline
default. The exception is the LLM: without it a live run stops and emails why, because a
script read out by a template from feed snippets isn't good enough to publish
(`SHORTS_ALLOW_NO_AI=on` overrides this; `run --offline` is unaffected).

## What never goes on air

The show is its own show, not somebody's newsletter. `content.py` holds the rules, and they
are checked in code, not just asked of the models:

- Source text is cleaned before any stage sees it: "This story originally appeared in our
  newsletter", "The post ... appeared first on ...", sign-up asks and forum points and
  comments are dropped.
- The script, headlines, key facts, title and description may not mention newsletters,
  Hacker News, Reddit, points, upvotes or comments, or talk like a publication ("our
  newsletter", "we reported"). The on-screen "via" line and "Covered by" credit only real
  publishers. The writer and critic are told the same, and a code check sends any slip back.
- Every story needs a real description: a story that is only a headline, or a link to a forum
  thread, is left out. A segment that repeats itself, barely goes beyond its headline, or
  copies 15+ words in a row from an article goes back for a rewrite.
- Duplicates are caught by headline comparison and then by a second model look.
- Fewer stories beats weak ones: up to `SHORTS_STORIES_PER_VIDEO` (8), at least
  `SHORTS_MIN_STORIES` (4), else no episode that day.
- The intro opens with a fresh hook each day (the last 7 intros are kept in
  `state/intros.json` and the writer must open differently). The outro is fixed:
  "Subscribe so you don't get lost in the storm of AI news. That's the news from the pond.
  See you tomorrow!" (`SHORTS_OUTRO`).

With an LLM key, the two judgement stages run as agents (`SHORTS_AGENTS=on`, the default):

- **Editor agent** (`selection.py`) can search the day's candidates and the list of stories
  that already aired while it picks. Its picks are then checked in code (`checks.py`): no
  duplicates, no repeats, only links that appear in the sources, nothing too old. Problems go
  back to it for up to `SHORTS_MAX_REPAIRS` rounds; picks that still fail are dropped, and
  only if fewer than `SHORTS_MIN_STORIES` remain is the list topped up from the keyword-scored
  pool, with stories whose feed text really describes them.
- **Writer agent** (`writer.py`) drafts the script, then a critic model checks every claim
  against the story material, and each segment's quality as TV news, while code checks length,
  hype words, links in the narration, numbers that aren't in the sources and the rules above.
  Only the failing segments are rewritten; any that still fail get the template line for that
  story, and a story whose template line would break the rules too is left out.

With `SHORTS_WEB=on` (web research, off by default until the shadow week below says it helps):

- The editor sees how widely the top headlines are covered right now (Hacker News points and
  the number of outlets on Google News), can look up any event's coverage, can search the web
  for a pick's primary source, and names up to 3 alternates.
- **Research agents** (`research.py`), one per story and all in parallel, read the article (or
  search for a better source), copy a few verbatim quotes and the date the event was first
  reported, and say whether the article is really about the pick. Code keeps only quotes it
  finds in the fetched pages and dates a page backs. A story that turns out wrong, stale or
  thin is swapped for an alternate (at most 2 a day) when the alternate checks out better.
- The writer and the critic work from those quotes instead of the whole article.

Pages are read with Tavily (counted against `SHORTS_TAVILY_MONTHLY_CREDITS`, kept in
`state/tavily.json`) or the free Jina reader when Tavily is off or out of credits.

Every LLM call is priced and logged in `cost.json`. A run stops calling the LLM at
`SHORTS_BUDGET_USD` (default $0.60) and the month at `SHORTS_MONTHLY_BUDGET_USD` (default $18,
kept in `state/spend.json`); stages then use their no-key fallbacks.

A run fails instead of uploading when every news source is down, when there's no LLM, when
there aren't enough solid stories, when the voice left a segment silent, or when a sample story
slipped in. The failure email says which.

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
| Read | Tavily extracts the full article for each pick; with `SHORTS_WEB=on`, a research agent per story checks it and quotes it | `TAVILY_API_KEY` (optional with `SHORTS_WEB=on`) | newsletter summary only | |
| Script | writer agent + critic (`SHORTS_WRITER_MODEL`, `SHORTS_CHECKER_MODEL`), persona in `shorts/persona.md` | LLM key | template line per failing segment | LangGraph newsroom with approval |
| Voice | edge-tts `en-US-AnaNeural` at +18% (free), or OpenAI voices (`SHORTS_VOICE=openai`); script trimmed if the audio runs past 170 s | internet (OpenAI: `OPENAI_API_KEY`) | silence, which fails QA | ElevenLabs designed voice |
| Host | puppet duck, bill moves with the voice loudness | – | – | Kling AI Avatar (`SHORTS_ANIMATOR`) |
| Render | cards + duck + desk + karaoke captions, one ffmpeg call | – | – | Remotion |
| Upload | `local` (metadata only) or `youtube` (private, marked synthetic) | YouTube OAuth secrets | – | public after API audit |
| Notify | email from the AgentMail inbox | `SHORTS_NOTIFY_EMAIL` | log line | Telegram approve button |

To add an implementation, write a class with the stage's method and register it in that
module's `build_*` function.

## Running daily on GitHub Actions

`.github/workflows/daily-short.yml` runs at 12:47 UTC and can be started by hand from the
Actions tab. It keeps the list of already-aired stories, the month's LLM spend and Tavily credits on a
`state` branch and saves the video and JSON files as a workflow artifact.

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
| `SHORTS_WEB` | variable | optional, `on` for web research (see below) |
| `SHORTS_SHADOW` | variable | optional, `on` for the shadow week (see below) |
| `SHORTS_TAVILY_MONTHLY_CREDITS`, `SHORTS_TAVILY_RUN_CREDITS` | variables | optional, default `700` and `25` |
| `SHORTS_UPLOADER` | variable | `youtube` once the YouTube secrets are in |
| `SHORTS_STORIES_PER_VIDEO`, `SHORTS_MIN_STORIES` | variables | optional, default `8` and `4` |
| `SHORTS_SOURCES` | variable | optional, default `newsletter,rss` |
| `SHORTS_VOICE`, `SHORTS_EDGE_VOICE`, `SHORTS_EDGE_RATE`, `SHORTS_OPENAI_VOICE`, `SHORTS_OPENAI_TTS_MODEL` | variables | optional; see "Choosing a voice" |
| `YOUTUBE_CLIENT_ID`, `YOUTUBE_CLIENT_SECRET`, `YOUTUBE_REFRESH_TOKEN` | secrets | see below |

## Choosing a voice

Run the workflow by hand (Actions, Daily AI Short, Run workflow) and fill in **voices** with
`all` (10 free Microsoft voices and 4 OpenAI voices), `edge`, `openai`, or a list like
`edge:en-GB-MaisieNeural,openai:coral`. After the episode is made, the same script is read by
each voice, and the run's download (`episode.zip`) has one MP3 per voice in `voices/` plus
`voices.txt` with the variables to set for each. The OpenAI voices cost a few cents per
comparison run. Locally: `python -m shorts voices --voices all` reads the last episode's script
(`state/last_episode.json`) or `--script output/<run>/03-episode.json`.

## Trying web research: the shadow week

Web research runs next to the live pipeline for a week before it airs anything:

1. Leave `SHORTS_WEB` unset and set the variable `SHORTS_SHADOW=on`.
2. Each daily run publishes the live episode as usual, then runs the web path on the same news
   into `output/<run>/shadow/` (never uploaded, never marks stories as aired) and emails a
   comparison: shared and different picks, what the researchers verified, how widely each
   side's picks are covered, one fact check that judges both scripts on the same quotes, QA,
   and cost. The shadow video is in the run's artifact; `state/shadow.jsonl` keeps a row a day.
3. The shadow skips itself when its cost could leave too little of the monthly budget for the
   live runs left this month. It has its own $0.60 run cap and counts toward the monthly cap.

A reasonable bar to switch: the shadow finished with 8 stories and passed QA on 6 of 7 days,
had no more unsupported claims than live on 5 days, averaged $0.50 or less, stayed on pace for
700 Tavily credits a month, and you would have aired its picks. To switch, set `SHORTS_WEB=on`
and remove `SHORTS_SHADOW`; to roll back, set `SHORTS_WEB=off`.

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
