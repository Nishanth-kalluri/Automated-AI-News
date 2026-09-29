"""What source text may reach the script, and what the script may say.

Feeds and newsletters carry text that is about the publication, not the news: "This story originally
appeared in The Algorithm, our weekly newsletter", "The post ... appeared first on ...", "412 points,
88 comments on HN". Read out by the host, that text makes the show sound like somebody else's
newsletter. ``clean_text`` strips it from everything the stages see, and ``script_problems`` checks
the script, headlines, title and description for it, so it can't come back through a model either.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from urllib.parse import urlsplit

from .models import Story

# The inbox's newsletters and other newsletters and aggregators that must never be credited on the show.
# Today's newsletter senders are added to these at run time (``banned_names``).
NEWSLETTER_NAMES = ("The Rundown", "Rundown AI", "TLDR", "Superhuman", "The Neuron", "The Algorithm",
                    "The Download", "The Batch", "Import AI", "Ben's Bites", "AI Breakfast", "AI newsletters")
# The same names as the host must never say them. Matched with their capitals and only in forms that
# can't be everyday words: "the algorithm", "here's the rundown", "import AI chips" and "superhuman
# performance" are normal lines, "The Algorithm" and "Superhuman AI" are newsletters.
_SPOKEN_NEWSLETTERS = [re.compile(p) for p in (
    r"\bThe Rundown(?: AI)?\b(?! on\b| of\b)", r"\bRundown AI\b", r"\bTLDR(?: AI)?\b", r"\bSuperhuman AI\b",
    # "The Algorithm" etc. are newsletters when used as a name, but a title-case headline ("The Algorithm
    # That Beat Go") is not one, so a capitalised word right after rules the match out.
    r"\bThe Neuron\b", r"\bThe (?:Algorithm|Download|Batch)\b(?! [A-Z])",
    r"\bImport AI\b(?! (?:chips?|models?|tools?|systems?|software|hardware))",
    r"\bBen's Bites\b", r"\bAI Breakfast\b",
)]
AGGREGATOR_NAMES = ("Hacker News", "Y Combinator News", "Reddit", "subreddit", "Techmeme")
# Links to discussion threads, not news.
AGGREGATOR_HOSTS = ("news.ycombinator.com", "reddit.com", "redd.it", "techmeme.com")
NEWSLETTER_HOSTS = ("therundown.ai", "tldr.tech", "superhuman.ai", "theneurondaily.com", "beehiiv.com",
                    "agentmail.to")

# Sentences in source text that are about the publication, not the news. Dropped whole.
_BOILERPLATE = [re.compile(p, re.I) for p in (
    r"\bnewsletters?\b",
    r"\bsign(?:ing)?[ -]up\b[^.!?]*\b(?:our|here|inbox|to get|for free)\b",
    r"\bsubscribe\b[^.!?]*\b(?:our|here|free|today|now|inbox|podcast|channel)\b",
    r"\b(?:subscribe|sign up) (?:here|now|today|for free)\b",
    r"\bin your inbox\b",
    r"\bappeared first (?:on|in)\b",
    r"\boriginally (?:appeared|published|ran)\b",
    r"\bclick here\b",
    r"\bread more\b(?! than)",
    r"\bread the (?:full|rest)\b",
    r"\bcontinue reading\b",
    r"(?<![-\w])sponsored (?:by|content|post|section|link)\b",
    r"\bpresented by\b",
    r"\badvertisement\b",
    r"\ball rights reserved\b",
    r"\bprivacy policy\b",
    r"\bterms of (?:service|use)\b",
    r"\bfollow us\b",
    r"\b\d[\d,]*\s+points?\b[^.!?]*\bcomments?\b",
    r"\b(?:on|via|from) (?:hn|hacker news|reddit)\b",
    r"\bupvot",
    r"\bthis (?:story|article|post) (?:was|is|first|originally)\b",
)]
_ELLIPSIS_MARK = re.compile(r"\s*(?:\[(?:…|\.\.\.)\]|\[&#8230;\])\s*")

# Things the host must never say or show. Checked in code on every spoken line, headline, key fact,
# the title and the description.
_SCRIPT_BANNED = [(re.compile(p, re.I), why) for p, why in (
    (r"\bnewsletters?\b", "mentions a newsletter"),
    (r"\boriginally (?:appeared|published|ran)\b", "says where the story originally appeared"),
    (r"\bappeared first\b", "says where the story first appeared"),
    (r"\bin your inbox\b", "talks about an inbox"),
    (r"\b\d[\d,.]*\s*k?\s+(?:upvotes|comments)\b", "reads out forum points or comments"),
    (r"\bpoints\b[^.!?]{0,40}\bcomments\b", "reads out forum points or comments"),
    (r"\bupvot", "reads out forum upvotes"),
    (r"\bcomment (?:section|thread)s?\b", "talks about a comment thread"),
    (r"\bhacker news\b|\bhn\b", "names Hacker News"),
    # News about Reddit the company is fine; Reddit as the source of a story isn't.
    (r"\b(?:on|from|via|over on|across) reddit\b|\breddit(?:ors?| users?| threads?| posts?| comments?| discussions?)\b"
     r"|\bsubreddits?\b|(?<![\w/])r/\w+", "names Reddit"),
    (r"\bour (?:weekly|daily|newsletter|reporting|reporters|readers|coverage|sister)\b", "speaks as a publication"),
    (r"\bwe (?:reported|wrote|covered|first reported)\b", "speaks as a publication"),
)]
# Only the outro may ask people to subscribe or follow.
_STORY_ONLY_BANNED = [(re.compile(r"\bsubscribe\b|\bfollow (?:us|for)\b", re.I), "asks people to subscribe")]

# Friendly publisher names by domain; the "via" line on screen and the writer's "Covered by" use these.
PUBLISHERS = {
    "techcrunch.com": "TechCrunch", "theverge.com": "The Verge", "technologyreview.com": "MIT Technology Review",
    "venturebeat.com": "VentureBeat", "arstechnica.com": "Ars Technica", "wired.com": "Wired",
    "reuters.com": "Reuters", "bloomberg.com": "Bloomberg", "wsj.com": "The Wall Street Journal",
    "nytimes.com": "The New York Times", "ft.com": "Financial Times", "cnbc.com": "CNBC", "axios.com": "Axios",
    "theinformation.com": "The Information", "engadget.com": "Engadget", "zdnet.com": "ZDNET",
    "businessinsider.com": "Business Insider", "fortune.com": "Fortune", "forbes.com": "Forbes",
    "theguardian.com": "The Guardian", "bbc.com": "BBC", "bbc.co.uk": "BBC", "cnn.com": "CNN",
    "washingtonpost.com": "The Washington Post", "semafor.com": "Semafor", "404media.co": "404 Media",
    "apnews.com": "AP", "theregister.com": "The Register", "9to5google.com": "9to5Google",
    "9to5mac.com": "9to5Mac", "macrumors.com": "MacRumors", "tomshardware.com": "Tom's Hardware",
    "openai.com": "OpenAI", "anthropic.com": "Anthropic", "blog.google": "Google", "google.com": "Google",
    "deepmind.google": "Google DeepMind", "deepmind.com": "Google DeepMind", "huggingface.co": "Hugging Face",
    "microsoft.com": "Microsoft", "nvidia.com": "Nvidia", "meta.com": "Meta", "about.fb.com": "Meta",
    "apple.com": "Apple", "amazon.com": "Amazon", "aboutamazon.com": "Amazon", "x.ai": "xAI",
    "mistral.ai": "Mistral AI", "amd.com": "AMD", "intel.com": "Intel", "ibm.com": "IBM",
    "arxiv.org": "arXiv", "github.com": "GitHub", "sec.gov": "SEC",
}
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
COPY_RUN_WORDS = 15  # this many words in a row, word for word from an article, is copying
MIN_DESCRIPTION_WORDS = 12  # a story needs at least this much real description to go on air
MIN_NEW_WORDS = 10  # of which this many must say more than the headline does


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def clean_text(text: str) -> str:
    """Source text without the sentences that are about the publication rather than the news."""
    text = _ELLIPSIS_MARK.sub(" ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    kept = [s for s in _sentences(text) if not any(p.search(s) for p in _BOILERPLATE)]
    return " ".join(kept).strip()


def _host(url: str) -> str:
    try:
        return (urlsplit(url or "").hostname or "").lower().removeprefix("www.")
    except ValueError:
        return ""


def _on(host: str, domains) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def is_aggregator_url(url: str) -> bool:
    return _on(_host(url), AGGREGATOR_HOSTS)


def is_banned_name(name: str, extra: tuple[str, ...] | set[str] = ()) -> bool:
    """Whether a source name is a newsletter or an aggregator, which the show never names."""
    low = (name or "").strip().lower()
    if not low:
        return False
    if re.match(r"r/\w+", low):
        return True
    return any(n.lower() in low for n in (*NEWSLETTER_NAMES, *AGGREGATOR_NAMES, *extra))


def publisher_name(url: str, feed_title: str = "") -> str:
    """The publisher to credit for a link: a known name, else the feed's own name, else the domain.

    Empty for newsletters, aggregators and anything without a usable link or name.
    """
    host = _host(url)
    if host and (_on(host, AGGREGATOR_HOSTS) or _on(host, NEWSLETTER_HOSTS)):
        return ""
    for domain, name in PUBLISHERS.items():
        if _on(host, (domain,)):
            return name
    if feed_title:
        # "AI News & Artificial Intelligence | TechCrunch", "Artificial intelligence – MIT Technology Review"
        name = re.split(r"\s+[|–—-]\s+", feed_title.strip())[-1].strip()
        if name and not is_banned_name(name):
            return name
    return host


def banned_names(candidates: list[Story]) -> set[str]:
    """Today's newsletter senders, on top of the known names."""
    return {c.source.strip() for c in candidates if c.kind == "newsletter" and c.source.strip()}


def script_problems(text: str, extra_names: set[str] | tuple[str, ...] = (), *, story: bool = True) -> list[str]:
    """What in ``text`` the host must never say: newsletters, forums, points and comments, talking like a
    publication. ``story`` also bans asking people to subscribe, which only the outro may do."""
    found = []
    for pattern, why in _SCRIPT_BANNED + (_STORY_ONLY_BANNED if story else []):
        if pattern.search(text or "") and why not in found:
            found.append(why)
    for name in extra_names:
        # Today's senders, with their capitals. A one-word sender ("Superhuman") is also an everyday
        # word or a company in the news, so only longer names are checked here; the patterns above
        # still catch the newsletter's own lines.
        if len(name.split()) > 1 and re.search(rf"(?<!\w){re.escape(name.strip())}(?!\w)", text or ""):
            found.append(f"names {name.strip()}")
    for pattern in _SPOKEN_NEWSLETTERS:
        m = pattern.search(text or "")
        if m and f"names {m.group(0)}" not in found:
            found.append(f"names {m.group(0)}")
    return found


def _words(text: str) -> list[str]:
    return _WORD.findall((text or "").lower().replace("’", "'"))


def repetition(text: str, headline: str = "") -> str:
    """Why a spoken segment repeats itself, or "": the same sentence twice, the headline read twice,
    or the same run of 6 words twice."""
    sentences = [" ".join(_words(s)) for s in _sentences(text or "")]
    sentences = [s for s in sentences if len(s.split()) >= 4]
    for i, a in enumerate(sentences):
        for b in sentences[i + 1:]:
            if SequenceMatcher(None, a, b).ratio() >= 0.8:
                return "says the same sentence twice"
    words = _words(text)
    head = _words(headline)
    if len(head) >= 3 and sum(words[i:i + len(head)] == head for i in range(len(words) - len(head) + 1)) >= 2:
        return "reads the headline twice"
    seen: set[tuple[str, ...]] = set()
    for i in range(len(words) - 5):
        gram = tuple(words[i:i + 6])
        if gram in seen:
            return f'repeats "{" ".join(gram)}"'
        seen.add(gram)
    return ""


def says_little(text: str, headline: str) -> bool:
    """A segment that barely goes beyond reading its headline."""
    head = set(_words(headline))
    new = [w for w in _words(text) if w not in head]
    return len(new) < MIN_NEW_WORDS


def copied_run(text: str, sources: list[str], run: int = COPY_RUN_WORDS) -> str:
    """The first run of ``run`` words that ``text`` copies word for word from one of ``sources``, or ""."""
    words = _words(text)
    if len(words) < run:
        return ""
    grams: set[tuple[str, ...]] = set()
    for src in sources:
        w = _words(src)
        grams.update(tuple(w[i:i + run]) for i in range(len(w) - run + 1))
    for i in range(len(words) - run + 1):
        if tuple(words[i:i + run]) in grams:
            return " ".join(words[i:i + run])
    return ""


def description_problem(story: Story) -> str:
    """Why a story can't go on air as it is, or "": it's a forum thread, or nothing describes it
    beyond the headline."""
    if is_aggregator_url(story.url):
        return "links to a discussion thread, not a news story"
    headline = story.headline or story.title
    if not headline.strip():
        return "has no headline"
    if story.evidence or len(clean_text(story.body).split()) >= 80:
        return ""  # the writer has checked quotes or the article itself
    summary = clean_text(story.summary)
    head = set(_words(headline))
    new = [w for w in _words(summary) if w not in head]
    if len(_words(summary)) < MIN_DESCRIPTION_WORDS or len(new) < MIN_NEW_WORDS:
        return "has no real description, only a headline"
    return ""
