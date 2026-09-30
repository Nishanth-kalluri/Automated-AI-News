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
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .models import Story

# The inbox's newsletters and other newsletters and aggregators that must never be credited on the show.
# Today's newsletter senders are added to these at run time (``banned_names``).
NEWSLETTER_NAMES = ("The Rundown", "Rundown AI", "TLDR", "Superhuman", "The Neuron", "The Algorithm",
                    "The Download", "The Batch", "Import AI", "Ben's Bites", "AI Breakfast", "AI newsletters")
# The same names as the host must never say them. Matched with their capitals and only in forms that
# can't be everyday words: "the algorithm", "here's the rundown", "import AI chips" and "superhuman
# performance" are normal lines, "The Algorithm" and "Superhuman AI" are newsletters.
_SPOKEN_NEWSLETTERS = [re.compile(p) for p in (
    r"\bThe Rundown(?: AI)?\b(?! on\b| of\b)", r"\bRundown AI\b", r"\bTLDR(?: AI)?\b",
    # "Superhuman AI could arrive by 2030" is about the technology, not the newsletter.
    r"\bSuperhuman AI\b(?! (?:could|can|will|would|may|might|is|isn't|was|by|within|arrives?|systems?|models?)\b)",
    # "The Algorithm" etc. are newsletters when used as a name, but a title-case headline ("The Algorithm
    # That Beat Go") is not one, so a capitalised word right after rules the match out.
    r"\bThe Neuron\b", r"\bThe (?:Algorithm|Download|Batch)\b(?! [A-Z])",
    r"\bImport AI\b(?! [A-Z]| (?:chips?|models?|tools?|systems?|software|hardware|chats?|data))",
    r"\bBen's Bites\b", r"\bAI Breakfast\b",
)]
AGGREGATOR_NAMES = ("Hacker News", "Y Combinator News", "Reddit", "subreddit", "Techmeme")
# Links to discussion threads, not news.
AGGREGATOR_HOSTS = ("news.ycombinator.com", "reddit.com", "redd.it", "techmeme.com")
NEWSLETTER_HOSTS = ("therundown.ai", "tldr.tech", "tldrnewsletter.com", "superhuman.ai", "theneurondaily.com",
                    "theneuron.ai", "beehiiv.com", "bensbites.com", "bensbites.co", "importai.substack.com", "jack-clark.net",
                    "agentmail.to")

# Sentences in source text that are about the publication, not the news. Dropped whole, so each
# pattern needs the publication talking about itself: "a bill sponsored by a senator", "Meta changed its
# privacy policy", "people who subscribe to Gemini" and "Gemini sorts your inbox" are news.
_SHORT = r"(?=\s*(?:\S+\s+){0,%d}\S*\s*$)"  # the rest of the sentence is at most this many words more
_BOILERPLATE = [re.compile(p, re.I) for p in (
    r"\b(?:our|this week's|today's)\s+(?:[\w'-]+\s+){0,2}newsletters?\b|\bthis newsletter\b"
    r"|\b(?:welcome to|thanks for reading)\s+(?:the\s+|this\s+)?(?:[\w'-]+\s+){0,2}newsletter\b",
    r"\b(?:subscribe|sign(?:ing)?[ -]up)\s+(?:to|for)\s+(?:our|this)\b",
    r"\b(?:subscribe|sign up)\s+(?:to|for)\s+the\b[^.!?]*\b(?:newsletter|podcast|channel|briefing|digest|feed)s?\b",
    r"\b(?:subscribe|sign up)(?: now| today)? here\b",
    r"^(?:please\s+)?(?:subscribe|sign up)\b" + _SHORT % 6,
    r"\bto get [^.!?]*\bin your inbox\b",
    r"\b(?:delivered|straight|directly|sent)\s+(?:to|in|into)\s+your inbox\b",
    r"\bin your inbox (?:first|every|each)\b",
    r"\bappeared first (?:on|in)\b",
    r"\boriginally (?:appeared|published|ran)\b",
    r"\bclick here\b",
    r"\bread more\b(?=\s*(?:at|here|[:\u00bb\u203a\u2192\u2026]|\.\.\.|[.!]?\s*$))"
    r"|\bread more (?:on|from) (?:(?-i:[A-Z])\w+|our|the site)\b",
    r"\bread (?:the )?(?:full|rest)\b",
    r"\bcontinue reading\b",
    r"^(?:this (?:[\w'-]+ )?(?:is |was )?)?sponsored\b" + _SHORT % 6,
    r"\bsponsored (?:content|post|section|link|message)\b",
    r"^(?:presented|brought to you) by\b" + _SHORT % 6 + r"|\bbrought to you by\b",
    r"^advertisement\b" + _SHORT % 4,
    r"\ball rights reserved\b",
    r"\bour (?:privacy policy|terms of (?:service|use))\b",
    r"\bfollow us\b",
    r"\b\d[\d,]*\s+points?\b[^.!?]{0,40}\bcomments?\b",
    r"\b(?:on|via|from) (?:hn|hacker news)\b",
    r"\b(?:discussed|trending|thread|discussion)\b[^.!?]{0,20}\b(?:on|over on) reddit\b",
    r"\bupvot",
    r"\bthis (?:story|article|post) (?:was|is|first|originally)\b",
)]
# Feed footers that are not whole sentences: "\u00a9 2026 TechCrunch. All rights reserved. For personal use
# only.", and Ars Technica's "Read full article Comments" links.
_FOOTER_SPANS = re.compile(r"(?:\u00a9|\bcopyright\b)\s*(?:\u00a9\s*)?\d{4}[^.]*\.?|\ball rights reserved\.?"
                           r"|\bfor personal use only\.?|\bread (?:the )?full (?:article|story)\b(?:\s+comments)?\s*$", re.I)
_ELLIPSIS_MARK = re.compile(r"\s*(?:\[(?:…|\.\.\.)\]|\[&#8230;\])\s*")

# Things the host must never say or show. Checked in code on every spoken line, headline, key fact,
# the title and the description.
_SCRIPT_BANNED = [(re.compile(p, re.I), why) for p, why in (
    (r"\bnewsletters?\b", "mentions a newsletter"),
    (r"\boriginally (?:appeared|published|ran)\b", "says where the story originally appeared"),
    (r"\bappeared first\b", "says where the story first appeared"),
    (r"\b(?:delivered|straight|directly|sent) (?:to|in|into) your inbox\b|\bin your inbox (?:first|every|each)\b"
     r"|\b(?:showed up|landed|arrived|came|hit|dropped) (?:in|into) (?:your|my|our) inbox\b|\bin (?:my|our) inbox\b",
     "talks about an inbox"),
    (r"\b\d[\d,.]*\s*k?\+?\s+upvotes\b", "reads out forum points or comments"),
    # Points and comments together, or a comment count on a thread or post. "10,000 comments on AI
    # training" (a regulator) and "the FTC points to public comments" are news.
    (r"\b\d[\d,.]*\s*k?\+?\s+points?\b[^.!?]{0,40}\bcomments\b|\b\d[\d,.]*\s*k?\+?\s+comments\b[^.!?]{0,40}\bpoints?\b"
     r"|\bpoints\b(?! (?:to|out)\b)[^.!?]{0,40}\bcomments\b"
     r"|\b\d[\d,.]*\s*k?\+?\s+comments\b[^.!?]{0,30}\b(?:hn|hacker news|reddit|thread|post)\b"
     r"|\b(?:thread|(?<!blog )post|discussion|forum|front page)s?\b[^.!?]{0,40}\b\d[\d,.]*\s*k?\+?\s+comments\b",
     "reads out forum points or comments"),
    # A key fact is its own line: "900 comments in a day" there has nothing else around it.
    (r"(?m)^\s*\d[\d,.]*\s*k?\+?\s+(?:comments|points|upvotes)\b", "reads out forum points or comments"),
    (r"\bupvot", "reads out forum upvotes"),
    (r"\bcomment (?:section|thread)s?\b", "talks about a comment thread"),
    (r"\bhacker news\b|\bhn\b", "names Hacker News"),
    # News about Reddit the company ("Google pays to train on Reddit posts, the companies said") is fine;
    # Reddit as where people said something isn't.
    (r"\bredditors?\b|\bsubreddits?\b|(?<![\w/])r/\w+"
     r"|\b(?:people|folks|users|commenters|fans|posters|developers|devs|engineers|researchers|critics|testers"
     r"|some|many|others|someone) (?:on|over on|across) reddit\b"
     r"|\b(?:on|over on|via|from) reddit\b(?=[^.!?]{0,40}\b(?:discussing|discussed|lov\w+|hat\w+|thinks?|thinking"
     r"|react\w*|jok\w+|went wild|upvot\w*|complain\w*|furious)\b)"
     r"|(?:^|[.!?]\s+)(?:over on|across|on|in) reddit\s*,|\bacross reddit\b|\baccording to (?:a |one )?reddit\b"
     r"|\b(?:a|an|one|this|that)\s+(?:viral\s+|popular\s+|top\s+)?reddit\s+(?:thread|post|user|comment|discussion)s?\b"
     r"|\b(?:viral|popular|top)\s+reddit\s+(?:threads?|posts?)\b"
     r"|\breddit (?:threads?|posts?|users?|comments?|commenters?|discussions?) (?:(?:are|were|is|was) (?:full|split"
     r"|divided|flooded|buzzing|calling|saying|claiming|complaining|sharing|furious|angry|convinced)\b"
     r"|(?:claim|say|said|found|find|discover|report|reveal|suggest|complain|call|argu|accus|think|love|hate|react)\w*)",
     "names Reddit"),
    (r"\bour (?:weekly|daily) (?:newsletter|edition|issue|roundup|briefing|digest)\b"
     r"|\bour (?:newsletter|reporting|reporters|readers|coverage|sister|weekly roundup)\b", "speaks as a publication"),
    (r"\bwe (?:reported|wrote|covered|first reported)\b", "speaks as a publication"),
)]
# Only the outro may ask people to subscribe or follow. "People who subscribe to ChatGPT Plus" is news.
_STORY_ONLY_BANNED = [(re.compile(
    r"(?:^|[.!?]\s+)(?:please |so |and )?(?:subscribe|follow)\b(?![-\w])"
    r"|\b(?:hit|smash|tap|click|press) (?:that |the )?subscribe\b"
    r"|\bfollow (?:us|for)\b|\bsubscribe (?:for (?:more|daily|tomorrow|the latest|updates|new)|to (?:the|this|our) "
    r"(?:channel|show))\b"
    # Any other "subscribe" except the news form: "people who subscribe to ChatGPT Plus", "subscribe for
    # 20 dollars a month".
    r"|(?<!\bwho )(?<!\bthat )(?<!\busers )\bsubscrib(?:e|ing)\b(?!\s+(?:to|monthly|annually|yearly)\b)"
    r"(?!\s+for\s+(?:\$|\d|about|around|under|just|only))", re.I | re.M),
    "asks people to subscribe")]

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
    text = re.sub(r"\s+", " ", _FOOTER_SPANS.sub(" ", text)).strip()
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


# Click-tracking hosts inside newsletters (links.tldr.tech, link.mail.beehiiv.com): they redirect to the
# real article, and research follows them.
_REDIRECT_LABELS = ("link", "links", "click", "clicks", "track", "tracking", "t", "r", "go", "l", "email", "mail")


def on_newsletter_host(url: str) -> bool:
    """Any link on a newsletter's domain, tracking redirects included: never shown or credited."""
    return _on(_host(url), NEWSLETTER_HOSTS)


def is_newsletter_url(url: str) -> bool:
    """A link to a newsletter's own web copy, which is never the story's source. A tracking redirect
    isn't one: it leads to the article."""
    host = _host(url)
    if not _on(host, NEWSLETTER_HOSTS):
        return False
    labels = host.split(".")
    return not (labels[0] in _REDIRECT_LABELS or "mail" in labels[1:-2])


def is_banned_name(name: str, extra: tuple[str, ...] | set[str] = ()) -> bool:
    """Whether a source name is a newsletter or an aggregator, which the show never names."""
    low = (name or "").strip().lower()
    if not low:
        return False
    if re.match(r"r/\w+", low):
        return True
    names = [n.lower() for n in (*NEWSLETTER_NAMES, *AGGREGATOR_NAMES, *extra) if n.strip()]
    # Also as a web address: "therundown.ai", "bensbites.com".
    squash = re.sub(r"[^a-z0-9]", "", low)
    return any(n in low or (len(re.sub(r"[^a-z0-9]", "", n)) >= 4 and re.sub(r"[^a-z0-9]", "", n) in squash)
               for n in names)


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


# Click and campaign ids that say where a reader came from ("?utm_source=tldrai"): dropped from any link
# viewers see, along with any parameter whose value names a newsletter or forum ("?ref=therundown").
_TRACKING_PARAMS = ("fbclid", "gclid", "mc_cid", "mc_eid", "_hsenc", "_hsmi", "mkt_tok", "oly_anon_id",
                    "oly_enc_id", "vero_id", "__s", "ck_subscriber_id")


def clean_url(url: str) -> str:
    """``url`` without tracking parameters, for the description and anything else viewers see."""
    try:
        parts = urlsplit((url or "").strip())
    except ValueError:
        return (url or "").strip()
    if not parts.query:
        return urlunsplit(parts)
    keep = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_PARAMS
            and not (v and is_banned_name(v.replace("_", " ").replace("-", " ")))]
    return urlunsplit(parts._replace(query=urlencode(keep)))


def credit(source: str, url: str) -> str:
    """The name to credit on screen: the story's source unless that is a newsletter, forum or a bare web
    address that isn't the link's own site (a tracking host kept from before research followed the
    link), in which case the link's publisher."""
    source = (source or "").strip()
    host = _host(url)
    bare = bool(re.fullmatch(r"[\w-]+(?:\.[\w-]+)+", source)) and not (host and _on(host, (source.lower(),)))
    if source and not bare and not is_banned_name(source):
        return source
    return publisher_name(url) if url else ""


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
            # "Sonnet costs 3 dollars ... Opus costs 15 dollars ..." are two facts, not one said twice.
            if re.findall(r"\d+(?:\.\d+)?", a) != re.findall(r"\d+(?:\.\d+)?", b):
                continue
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


def shared_sentence(text: str, other: str, min_words: int = 3) -> str:
    """A sentence of ``other`` (at least ``min_words`` words) that ``text`` says word for word, or ""."""
    said = f" {' '.join(_words(text))} "
    for sentence in _sentences(other or ""):
        words = _words(sentence)
        if len(words) >= min_words and f" {' '.join(words)} " in said:
            return sentence
    return ""


def says_little(text: str, headline: str) -> bool:
    """A segment that barely goes beyond reading its headline."""
    head = set(_words(headline))
    new = [w for w in _words(text) if w not in head]
    return len(new) < MIN_NEW_WORDS


def shared_run(text: str, other: str, n: int = 5, skip: set[str] | frozenset[str] = frozenset()) -> str:
    """The first ``n`` words in a row that both texts say, with at least two of them outside ``skip``
    (filler and names, which any two lines about the same story share)."""
    a = [w.removesuffix("'s") for w in _words(text)]
    b = [w.removesuffix("'s") for w in _words(other)]
    grams = {tuple(b[i:i + n]) for i in range(len(b) - n + 1)}
    for i in range(len(a) - n + 1):
        gram = tuple(a[i:i + n])
        if gram in grams and sum(w not in skip for w in gram) >= 2:
            return " ".join(gram)
    return ""


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
