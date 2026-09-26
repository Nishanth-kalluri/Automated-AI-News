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


def _event_words(text: str) -> set[str]:
    """The words that identify an event: names, products, numbers.

    Numbers are compared by value, however they are written: "GPT-5" and "GPT 5" give "5",
    "$100B" and "$100 billion" give "100", "Gemini 3.0" gives "3".
    """
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", (text or "").lower())  # typographic hyphens
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    words = set()
    for w in re.findall(r"\d+(?:\.\d+)?[a-z]*|[a-z][a-z0-9]*", text):
        number = _EVENT_NUMBER_RE.fullmatch(w)
        if number:
            w = f"{float(number.group(1)):g}"
        elif w in _FILLER or len(w) < 2:
            continue
        else:
            w = w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
        words.add(w)
    return words - _FILLER


def similar(a: str, b: str) -> float:
    """Share of event words two headlines have in common (0 to 1)."""
    wa, wb = _event_words(a), _event_words(b)
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


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
    # "Anthropic raises $13B at $183B valuation" / "Anthropic raises $13 billion Series F, valued at $183 billion"
    same_amount = any(w[0].isdigit() and float(w) >= 10 for w in shared)
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
                now: datetime | None = None) -> list[Issue]:
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
        if not url:
            issues.append(Issue("no_url", f"{name!r} has no link", i, fatal=False))
        elif url not in known_urls:
            issues.append(Issue("url_not_in_sources", f"the link {s.url} does not appear in today's material", i))
        age_h = (now - s.published).total_seconds() / 3600
        if age_h > max_age_hours:
            issues.append(Issue("stale", f"{name!r} was published {age_h:.0f} hours ago", i))
        if live_candidates and is_sample(s):
            issues.append(Issue("sample", f"{name!r} is a built-in sample story, not real news", i))
    if len(picks) < n:
        issues.append(Issue("too_few", f"only {len(picks)} of {n} stories"))
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
                     *story.outlets])


def episode_material(stories: list[Story], frame: str = "") -> str:
    """Everything the intro, outro and title may draw on: every story, the story count, the frame."""
    return " ".join(_material(s) for s in stories) + f" {len(stories)} {max(len(stories) - 1, 0)} {frame}"


def _speech_problems(text: str) -> list[str]:
    lowered = text.lower()
    problems = []
    hype = [w for w in HYPE_WORDS if w in lowered]
    if hype:
        problems.append(f"uses hype words ({', '.join(hype)})")
    if _URL_RE.search(text) or re.search(r"\bwww\.", lowered):
        problems.append("reads out a web address")
    return problems


def lint_episode(episode: Episode, stories: list[Story], frame: str = "") -> list[Issue]:
    """Script rules the writer is told about, checked in code rather than trusted.

    ``frame`` is what else the intro and outro may mention: the show, the host and today's date.
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
    for code, where, seg in (("intro_problem", "the intro", episode.segments[0]),
                             ("outro_problem", "the outro", episode.segments[-1])):
        if seg.kind not in ("intro", "outro") or not seg.text.strip():
            continue
        problems = _speech_problems(seg.text)
        missing = unsupported_numbers(seg.text, everything)
        if missing:
            problems.append(f"uses {', '.join(missing)}, which is not in the story material")
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
        missing = unsupported_numbers(f"{seg.text}\n{seg.key_fact}\n{seg.headline}", _material(story))
        if missing:
            issues.append(Issue("unsupported_number",
                                f"uses {', '.join(missing)}, which is not in the story material", i))
        if len(seg.headline.split()) > 8:
            issues.append(Issue("long_headline", "headline is longer than 8 words", i, fatal=False))
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
