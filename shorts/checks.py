"""Rule checks shared by the pipeline and the agents.

Every check is a plain function that returns a list of ``Issue``s. The pipeline uses them as
guards, and the agent loops use the same issues to decide what to send back for repair, so
a problem is described once and fixed by whichever stage can fix it.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .content import (copied_run, is_aggregator_url, is_newsletter_url, repetition, says_little, script_problems,
                      shared_sentence)
from .models import Episode, Story

SAMPLE_URL_PREFIX = "https://example.com/sample/"
STORY_WORDS = (30, 42)  # ~15 seconds of speech per story segment
WORDS_PER_SECOND = 2.6  # same pace the silent voice uses
SEGMENT_GAP = 0.3
TARGET_MAX_SECONDS = 170  # leaves headroom under the 180 s Shorts limit
HYPE_WORDS = ("revolutionary", "game-changer", "game changer", "mind-blowing", "mind blowing")
TITLE_MAX = 91  # the uploader appends " #Shorts" and YouTube allows 100 characters

_URL_RE = re.compile(r"https?://[^\s)>\]\"'<]+")
# A number as written, with an optional scale word: "93.4", "1,500,000", "$1.49 billion", "40k".
_NUM_RE = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?:\s*(thousand|million|billion|trillion|bn|tn|[kmbt])\b)?",
                     re.I)
_SCALES = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9,
           "trillion": 1e12, "tn": 1e12, "t": 1e12}
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "dozen": 12, "fifteen": 15, "twenty": 20,
                 "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100, "half": 0.5, "double": 2, "twice": 2,
                 "triple": 3, "quarter": 0.25}
# Words that say nothing about which event a headline is about; left out when comparing headlines.
_FILLER = set("""a an the and or but of for to in on at by with from as is are was were be been being its it this that
these those new now just ai how why what who says said say will can could may might more most over into after
about than you your we our their they his her he she has have had not no gets get got adds add added makes make
made launch launches launched unveil unveils unveiled release releases released announce announces announced
ship ships shipped introduce introduces introduced debut debuts roll rolls rolled out rolling raise raises raised
open opens opened build builds built model models tool tools feature features update updates version company
startup lab labs report reports study billion million thousand dollar dollars percent funding round first big
major latest today week again also here plus up bring brings brought sign signs signed hire hires hired acquire
acquires acquired buy buys bought invest invests invested partner partners partnered""".split())
_EVENT_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)(?:bn|tn|[kmbt])?")


@dataclass
class Issue:
    code: str
    detail: str
    index: int | None = None  # 0-based story position, or None for the whole pick list / episode
    fatal: bool = True  # False: worth fixing, fine to ship if it can't be

    def line(self) -> str:
        where = f"story {self.index + 1}: " if self.index is not None else ""
        return f"- {where}{self.detail}"


def is_sample(story: Story) -> bool:
    return story.url.startswith(SAMPLE_URL_PREFIX)


def norm_url(url: str) -> str:
    """Comparable form of a URL: lower-case host, no tracking params, fragment or trailing slash.

    Empty for anything that isn't a parseable http(s) link.
    """
    url = (url or "").strip()
    if not url.startswith("http"):
        return ""
    try:
        parts = urlsplit(url)
    except ValueError:  # e.g. "https://[UNSUBSCRIBE]" in a newsletter, or a bare IPv6 host
        return ""
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if not k.lower().startswith("utm_")])
    path = parts.path.rstrip("/")
    return urlunsplit(("https", parts.netloc.lower().removeprefix("www."), path, query, ""))


def _event_list(text: str) -> list[str]:
    """``_event_words`` in reading order, each once."""
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", (text or "").lower())  # typographic hyphens
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    words: list[str] = []
    for w in re.findall(r"\d+(?:\.\d+)?[a-z]*|[a-z][a-z0-9]*", text):
        number = _EVENT_NUMBER_RE.fullmatch(w)
        if number:
            w = f"{float(number.group(1)):g}"
        elif w in _FILLER or len(w) < 2:
            continue
        else:
            w = w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
        if w not in _FILLER and w not in words:
            words.append(w)
    return words


def _event_words(text: str) -> set[str]:
    """The words that identify an event: names, products, numbers.

    Numbers are compared by value, however they are written: "GPT-5" and "GPT 5" give "5",
    "$100B" and "$100 billion" give "100", "Gemini 3.0" gives "3".
    """
    return set(_event_list(text))


def _first_event_word(text: str) -> str:
    """The first event word in reading order: usually who did it ("Nvidia launches ...")."""
    words = _event_list(text)
    return words[0] if words else ""


def _shared_after_name(text: str, shared: set[str]) -> int:
    """How many shared event words come after the headline's opening run of shared words.

    The opening run is who and what it is about ("Meta Ray-Ban Display glasses"); what happened comes
    after it, so two headlines that share only the opening run are different news about one product.
    """
    words = _event_list(text)
    i = 0
    while i < len(words) and words[i] in shared:
        i += 1
    return sum(w in shared for w in words[i:])


def similar(a: str, b: str) -> float:
    """Share of event words two headlines have in common (0 to 1)."""
    wa, wb = _event_words(a), _event_words(b)
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def _amount(word: str) -> float:
    """An event word's value; 0 for words that only start with a digit, like "2nm", "4o" or "10x"."""
    try:
        return float(word)
    except ValueError:
        return 0.0


def same_event(a: str, b: str) -> bool:
    """Whether two headlines are about the same event.

    Headlines follow a few patterns ("X unveils Y", "X signs a deal with Y"), so they are compared on
    their event words only. Two headlines with different numbers are different events, and so are
    two that differ in the thing that happened: "Nvidia unveils new AI chip" and "AMD unveils new
    AI chip", or "OpenAI signs chip deal with AMD" and "OpenAI signs chip deal with Broadcom".
    """
    wa, wb = _event_words(a), _event_words(b)
    nums_a = {w for w in wa if w[0].isdigit()}
    nums_b = {w for w in wb if w[0].isdigit()}
    if nums_a and nums_b and not nums_a & nums_b:
        return False
    shared = wa & wb
    if len(shared) < 2:  # short headlines like "OpenAI model launch" and "OpenAI model launch again"
        return bool(wa) and wa == wb
    if shared in (wa, wb) or len(shared) / len(wa | wb) >= 0.75:
        return True
    # "Nvidia launches new platform for reining in rogue AI agents" / "Nvidia says its new AI safety
    # platform can contain rogue agents within 'milliseconds'": four names and nouns in common, and the
    # same company doing it. "Microsoft launches ..." and "Nvidia launches ..." the same thing are two events.
    # "Meta's Ray-Ban Display glasses fail live demo" / "... go on sale" share only the product's name.
    actor_a, actor_b = _first_event_word(a), _first_event_word(b)
    if (len(shared) >= 4 and len(shared) / min(len(wa), len(wb)) >= 0.7
            and actor_a in wb and actor_b in wa
            and _shared_after_name(a, shared) >= 2 and _shared_after_name(b, shared) >= 2):
        return True
    # "Anthropic raises $13B at $183B valuation" / "Anthropic raises $13 billion Series F, valued at $183 billion"
    same_amount = any(w[0].isdigit() and _amount(w) >= 10 for w in shared)
    return same_amount and len(shared) / min(len(wa), len(wb)) >= 0.75


def source_urls(candidates: list[Story]) -> set[str]:
    """Every URL the day's material mentions: feed links plus links inside newsletter text."""
    urls = {norm_url(c.url) for c in candidates if c.url}
    for c in candidates:
        urls |= {norm_url(u.rstrip(".,;:")) for u in _URL_RE.findall(f"{c.body} {c.summary}")}
    urls.discard("")
    return urls


def check_picks(picks: list[Story], candidates: list[Story], n: int, seen_urls: set[str],
                aired_headlines: list[str], max_age_hours: float,
                now: datetime | None = None, min_n: int | None = None) -> list[Issue]:
    """``n`` is how many stories to aim for; fewer than ``min_n`` (default ``n``) is a problem."""
    now = now or datetime.now(timezone.utc)
    known_urls = source_urls(candidates)
    aired_urls = {norm_url(u) for u in seen_urls} - {""}
    live_candidates = any(not is_sample(c) for c in candidates)
    issues: list[Issue] = []
    for i, s in enumerate(picks):
        name = s.headline or s.title
        if not (name and s.summary):
            issues.append(Issue("incomplete", "missing a headline or summary", i))
            continue
        url = norm_url(s.url)
        for j, other in enumerate(picks[:i]):
            if (url and url == norm_url(other.url)) or same_event(name, other.headline or other.title):
                issues.append(Issue("duplicate", f"same event as story {j + 1} ({other.headline or other.title!r})", i))
                break
        if url and url in aired_urls:
            issues.append(Issue("already_aired", f"{name!r} already aired (same link)", i))
        else:
            match = next((h for h in aired_headlines if same_event(name, h)), None)
            if match:
                issues.append(Issue("already_aired", f"{name!r} looks like {match!r}, which already aired", i))
        if is_aggregator_url(s.url):
            issues.append(Issue("not_news", f"{name!r} links to a discussion thread, not a news story; "
                                            "use the original article or pick another story", i))
        elif is_newsletter_url(s.url):
            issues.append(Issue("not_news", f"{name!r} links to a newsletter's own page; use the original "
                                            "article's link, or leave the link empty", i))
        if not url:
            issues.append(Issue("no_url", f"{name!r} has no link", i, fatal=False))
        elif url not in known_urls:
            issues.append(Issue("url_not_in_sources", f"the link {s.url} does not appear in today's material", i))
        age_h = (now - s.published).total_seconds() / 3600
        if age_h > max_age_hours:
            issues.append(Issue("stale", f"{name!r} was published {age_h:.0f} hours ago", i))
        if live_candidates and is_sample(s):
            issues.append(Issue("sample", f"{name!r} is a built-in sample story, not real news", i))
    need = n if min_n is None else min_n
    if len(picks) < need:
        issues.append(Issue("too_few", f"only {len(picks)} stories; at least {need} different, solid news events "
                                       f"are needed (up to {n})"))
    return issues


def _numbers(text: str) -> list[tuple[str, float, int]]:
    """Every number in ``text`` as (how it was written, value with its scale word applied, decimals)."""
    found = []
    for m in _NUM_RE.finditer(text or ""):
        digits, scale = m.group(1).replace(",", "").rstrip("."), (m.group(2) or "").lower()
        try:
            value = float(digits) * _SCALES.get(scale, 1)
        except ValueError:
            continue
        found.append((m.group(0).strip(), value, len(digits.partition(".")[2])))
    return found


def _decimal_points(text: str) -> str:
    return re.sub(r"\b(\d+) point (\d+)\b", r"\1.\2", text or "")


def _material_values(text: str) -> set[float]:
    """Numbers a script may use: each number with and without its scale, and numbers written as
    words ("three models", "a million-token context", "one billion users")."""
    values: set[float] = set()
    for m in _NUM_RE.finditer(_decimal_points(text)):
        digits = m.group(1).replace(",", "").rstrip(".")
        try:
            values.add(float(digits))
            values.add(float(digits) * _SCALES.get((m.group(2) or "").lower(), 1))
        except ValueError:
            continue
    low = (text or "").lower()
    values.update(v for w, v in _NUMBER_WORDS.items() if re.search(rf"\b{w}\b", low))
    words = "|".join(["an?", *_NUMBER_WORDS])
    for word, scale in re.findall(rf"\b({words})[\s-]+(thousand|million|billion|trillion)\b", low):
        values.add(_NUMBER_WORDS.get(word, 1) * _SCALES[scale])
    return values


def _round_sig(x: float, digits: int) -> float:
    return 0.0 if x == 0 else round(x, digits - 1 - int(math.floor(math.log10(abs(x)))))


def _supported(value: float, decimals: int, material: set[float]) -> bool:
    """Exact, or the material number rounded to 2-3 significant figures or to the script's decimals:
    "93 percent" for 93.4%, "1.5 billion" for $1.49 billion. A different year is never a rounding."""
    for m in material:
        if math.isclose(value, m, rel_tol=1e-9):
            return True
        if 1900 <= value <= 2100 and float(value).is_integer():
            continue  # looks like a year: exact only
        if any(math.isclose(value, _round_sig(m, d), rel_tol=1e-9) for d in (2, 3)):
            return True
        # Rounding to whole numbers only from 10 up: "Claude 5" is not a rounding of "Claude 4.5".
        if abs(m) >= 10 and math.isclose(value, round(m, decimals), rel_tol=1e-9):
            return True
    return False


def unsupported_numbers(text: str, material: str) -> list[str]:
    """Numbers in ``text`` the material doesn't back, as written. "4 point 1" reads as 4.1."""
    values = _material_values(material)
    return sorted({written for written, value, decimals in _numbers(_decimal_points(text))
                   if not _supported(value, decimals, values)})


def predicted_seconds(episode: Episode) -> float:
    words = sum(len(s.text.split()) for s in episode.segments)
    return words / WORDS_PER_SECOND + SEGMENT_GAP * max(len(episode.segments) - 1, 0)


def _material(story: Story) -> str:
    return " ".join([story.title, story.headline, story.summary, story.body, story.key_fact, story.source,
                     *story.outlets, *(e.quote for e in story.evidence)])


def episode_material(stories: list[Story], frame: str = "") -> str:
    """Everything the intro, outro and title may draw on: every story, the story count, the frame."""
    return " ".join(_material(s) for s in stories) + f" {len(stories)} {max(len(stories) - 1, 0)} {frame}"


def opening(text: str, words: int = 4) -> str:
    """How a line opens, for telling whether two intros start the same way: the first words, no punctuation."""
    return " ".join(re.findall(r"[a-z0-9']+", (text or "").lower().replace("\u2019", "'"))[:words])


def _speech_problems(text: str, banned: set[str] | tuple[str, ...] = (), *, outro: bool = False) -> list[str]:
    """Only the outro may ask people to subscribe; it already does, so the intro mustn't."""
    lowered = text.lower()
    problems = list(script_problems(text, banned, story=not outro))
    hype = [w for w in HYPE_WORDS if w in lowered]
    if hype:
        problems.append(f"uses hype words ({', '.join(hype)})")
    if _URL_RE.search(text) or re.search(r"\bwww\.", lowered):
        problems.append("reads out a web address")
    return problems


def lint_episode(episode: Episode, stories: list[Story], frame: str = "", *,
                 banned: set[str] | tuple[str, ...] = (), recent_intros: list[str] | tuple[str, ...] = (),
                 description: str | None = None) -> list[Issue]:
    """Script rules the writer is told about, checked in code rather than trusted.

    ``frame`` is what else the intro and outro may mention: the show, the host and today's date.
    ``banned`` adds today's newsletter names to the ones the show never says. ``recent_intros`` are
    the last episodes' intros, which today's must not open like. ``description`` is the writer's own
    YouTube description, without the sources footer.
    """
    issues: list[Issue] = []
    kinds = [s.kind for s in episode.segments]
    if "intro" not in kinds or not episode.segments[0].text.strip():
        issues.append(Issue("missing_intro", "the intro is missing"))
    if "outro" not in kinds or not episode.segments[-1].text.strip():
        issues.append(Issue("missing_outro", "the outro is missing"))
    segs = episode.story_segments
    if len(segs) != len(stories):
        issues.append(Issue("segment_count", f"{len(segs)} story segments for {len(stories)} stories"))
    everything = episode_material(stories, frame)
    # The show's own sign-off, which only the outro says.
    signoff = episode.segments[-1].text if episode.segments and episode.segments[-1].kind == "outro" else ""
    for code, where, seg in (("intro_problem", "the intro", episode.segments[0]),
                             ("outro_problem", "the outro", episode.segments[-1])):
        if seg.kind not in ("intro", "outro") or not seg.text.strip():
            continue
        problems = _speech_problems(seg.text, banned, outro=seg.kind == "outro")
        missing = unsupported_numbers(seg.text, everything)
        if missing:
            problems.append(f"uses {', '.join(missing)}, which is not in the story material")
        if seg.kind == "intro":
            start = opening(seg.text)
            if start and any(opening(old) == start for old in recent_intros):
                problems.append(f'opens with "{start}", like a recent episode; use a fresh hook')
            if shared_sentence(seg.text, signoff):
                problems.append("says the show's sign-off, which only the outro says")
        if problems:
            issues.append(Issue(code, f"{where} {'; '.join(problems)}"))
    for i, (seg, story) in enumerate(zip(segs, stories)):
        words = len(seg.text.split())
        if words > STORY_WORDS[1] + 6:
            issues.append(Issue("too_long", f"{words} words; keep it to {STORY_WORDS[0]}-{STORY_WORDS[1]}", i))
        elif words < STORY_WORDS[0] - 8:
            issues.append(Issue("too_short", f"only {words} words; aim for {STORY_WORDS[0]}-{STORY_WORDS[1]}", i,
                                fatal=False))
        lowered = seg.text.lower()
        hype = [w for w in HYPE_WORDS if w in lowered]
        if hype:
            issues.append(Issue("hype", f"uses hype words ({', '.join(hype)})", i))
        if _URL_RE.search(seg.text) or re.search(r"\bwww\.", lowered):
            issues.append(Issue("url_in_narration", "reads out a web address", i))
        talk = script_problems(f"{seg.text}\n{seg.headline}\n{seg.key_fact}", banned)
        if talk:
            issues.append(Issue("source_talk", "; ".join(talk) + ". Tell the news itself, as the show's host", i))
        repeated = repetition(seg.text, seg.headline)
        if not repeated and shared_sentence(seg.text, signoff):
            repeated = "says the show's sign-off, which the outro already says"
        if repeated:
            issues.append(Issue("repeats", f"{repeated}; say each thing once", i))
        elif says_little(seg.text, seg.headline):
            issues.append(Issue("says_little", "barely goes beyond the headline; say what happened, with the specifics",
                                i))
        # A story picked from the keyword ranking (it has a score) keeps the feed's own summary: the
        # outlet's words, which the show must not read out either.
        feed_summary = [story.summary] if story.score > 0 else []
        copied = copied_run(seg.text, [story.body, *(e.quote for e in story.evidence), *feed_summary])
        if copied:
            issues.append(Issue("copied", f'copies the article word for word ("{copied}"); say it in your own words',
                                i))
        missing = unsupported_numbers(f"{seg.text}\n{seg.key_fact}\n{seg.headline}", _material(story))
        if missing:
            issues.append(Issue("unsupported_number",
                                f"uses {', '.join(missing)}, which is not in the story material", i))
        if len(seg.headline.split()) > 8:
            issues.append(Issue("long_headline", "headline is longer than 8 words", i, fatal=False))
    title_talk = script_problems(episode.title, banned)
    if title_talk:
        issues.append(Issue("title_problem", f"the title {'; '.join(title_talk)}"))
    if description is not None:
        desc_talk = script_problems(description, banned)
        if desc_talk:
            issues.append(Issue("description_problem", f"the description {'; '.join(desc_talk)}"))
    if len(episode.title) > TITLE_MAX:
        issues.append(Issue("long_title", f"title is {len(episode.title)} characters; keep it under {TITLE_MAX}",
                            fatal=False))
    seconds = predicted_seconds(episode)
    if seconds > TARGET_MAX_SECONDS:
        extra = int((seconds - TARGET_MAX_SECONDS) * WORDS_PER_SECOND) + 1
        issues.append(Issue("too_long_total", f"the episode runs about {seconds:.0f}s; cut about {extra} words"))
    return issues


def fatal(issues: list[Issue]) -> list[Issue]:
    return [i for i in issues if i.fatal]
