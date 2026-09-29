"""Stage 4: turn the picked stories into an episode: intro, one segment per story, outro.

``LLMWriter`` makes one call. ``CriticWriter`` (the agent) drafts with it, checks the script in
code and with a fact-checking critic, and sends only the failing segments back for rewriting.
Whatever still fails after the repair rounds is replaced segment by segment with template
text, so one bad segment never costs the whole script.
"""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from .checks import (STORY_WORDS, TARGET_MAX_SECONDS, WORDS_PER_SECOND, Issue, _event_words, episode_material,
                     fatal, lint_episode, opening, predicted_seconds, unsupported_numbers)
from .config import DEFAULT_OUTRO
from .content import (clean_text, is_aggregator_url, is_banned_name, is_newsletter_url, repetition,
                      script_problems)
from .llm import LLM, BudgetExceeded, strict_object
from .models import Episode, Segment, Story

log = logging.getLogger(__name__)
PERSONA_FILE = Path(__file__).with_name("persona.md")
SIGN_OFF = "That's the news from the pond. See you tomorrow!"
# Openings for the template intro, one per day, so even the fallback doesn't sound the same every day.
TEMPLATE_HOOKS = (
    "Quack quack, it's {host} on {show}!",
    "Waddle in, friends, it's {host} on {show}!",
    "Quack! {host} here, and the AI pond is busy today.",
    "Ruffle those feathers, it's {host} on {show}!",
    "Fresh from the pond, it's {host} on {show}!",
    "Splash! {host} here with today's {show}.",
    "Quack attack! It's {host} on {show}.",
)
# Problems about the intro or outro only; they don't open the story segments for rewriting.
FRAME_CODES = ("missing_intro", "missing_outro", "intro_problem", "outro_problem")
# Problems only settle() fixes, by using the standard title or description.
SETTLE_ONLY = ("title_problem", "description_problem")

WRITER_SYSTEM = """You write the script for a daily vertical YouTube Short of about 2 minutes.

{persona}

Format:
- "intro": 12 to 20 words. Open with a fresh, playful hook in the host's voice that grabs attention in the first
  second: a quack, a duck pun, a surprising fact from the biggest story, or a question. Then tease the biggest
  story. Every day's hook is new: never open like one of the recent intros you are shown. Don't count the stories.
- "segments": exactly one per story, in the given order. Each is {lo} to {hi} words (about 15 seconds spoken):
  what happened, with the key specifics (who, what, how much, when), then one short line on why it matters.
  Describe the news; never just read the headline out, and never say the same thing twice.
- "outro": always "". The show adds its own outro.
- Written to be heard: short sentences, no parentheses, no URLs, no emoji, spell out symbols ("percent", "dollars").
- Only use facts given in the story material. Never invent numbers, names or quotes.
- Say it in your own words. Never read out more than a short phrase word for word from an article.
- You are the host of this show, not a newsletter or a news site. Never mention newsletters or Hacker News,
  never say what people on Reddit or in comments said, never read out points, upvotes or comment counts, and
  never say things like "our newsletter", "we reported" or "this story originally appeared". News about Reddit
  the company is fine. When a source is worth naming, name the original publisher: "according to The Verge".
- The title, description, intro and segments never ask viewers to subscribe, like or follow, and never say the
  show's sign-off: the show's own outro does both.

Return JSON:
{{"title": "YouTube title under 70 characters, no hashtags",
  "description": "2 to 3 sentences describing today's episode",
  "tags": ["5 to 10 tags"],
  "intro": "...",
  "segments": [{{"headline": "on-screen headline, max 8 words", "key_fact": "max 7 words or empty", "text": "..."}}],
  "outro": ""}}"""

SEGMENT_SCHEMA = strict_object({"headline": {"type": "string"}, "key_fact": {"type": "string"},
                                "text": {"type": "string"}})
SCRIPT_SCHEMA = strict_object({
    "title": {"type": "string"},
    "description": {"type": "string"},
    "tags": {"type": "array", "items": {"type": "string"}},
    "intro": {"type": "string"},
    "segments": {"type": "array", "items": SEGMENT_SCHEMA},
    "outro": {"type": "string"},
})
REVISION_SCHEMA = strict_object({
    "intro": {"type": "string"},
    "outro": {"type": "string"},
    "segments": {"type": "array", "items": strict_object({
        "story": {"type": "integer", "description": "story number, starting at 1"},
        "headline": {"type": "string"}, "key_fact": {"type": "string"}, "text": {"type": "string"}})},
})
REVIEW_SCHEMA = strict_object({
    "intro": {"type": "array", "items": {"type": "string"},
              "description": "claims in the intro that the material does not support"},
    "outro": {"type": "array", "items": {"type": "string"},
              "description": "always empty: the outro is the show's fixed sign-off"},
    "segments": {"type": "array", "items": strict_object({
        "story": {"type": "integer", "description": "story number, starting at 1"},
        "unsupported": {"type": "array", "items": {"type": "string"},
                        "description": "claims in the segment or its key fact that the material does not support"},
        "quality": {"type": "array", "items": {"type": "string"},
                    "description": "quality problems: headline only, repeats, names a newsletter or forum, "
                                   "talks like a publication, hard to follow"}})},
})

CRITIC_SYSTEM = """You fact-check the script of a daily AI news Short against the source material for each story.
For every story segment, list each claim (a name, number, date, quote or event) in the segment text or its key
fact that the material does not support. Check the intro the same way against all the stories. The outro is
the show's fixed sign-off: leave its list empty.
Paraphrase and rounding are fine; new facts, wrong numbers and claims about the wrong company are not.

Then list each segment's quality problems as TV news, under "quality":
- it only restates the headline and never says what actually happened;
- it repeats itself;
- it mentions a newsletter or Hacker News, says what people on Reddit or in comments said, or reads out
  points, upvotes or comment counts (news about Reddit the company is fine);
- it talks as if the show were a newsletter or news site ("our newsletter", "we reported", "originally appeared");
- a viewer couldn't follow it.
Return empty lists for a segment with no problems."""

CRITIC_EVIDENCE_NOTE = """
Some stories list Evidence (verbatim quotes from the sources) and a First reported date. For those, check the
segment, headline and key fact against the Evidence first: a claim is supported only if the Evidence, the
Summary or the Article text (when given) states it, and where they disagree the Evidence wins. "Today",
"yesterday" or "this week" in conflict with the First reported date is unsupported. Check stories without
Evidence as before."""

REVISE_PROMPT = """Here is today's script and the story material. Fix only the problems listed below.

=== PROBLEMS ===
{problems}

=== CURRENT SCRIPT ===
{script}

=== STORY MATERIAL ===
{material}

Return JSON with "intro", "outro" (unchanged unless a problem is about them) and "segments" holding only
the segments you changed, each with its story number. Keep every rule from your instructions."""


def load_persona(show: str, host: str) -> str:
    return PERSONA_FILE.read_text().format(show=show, host=host)


class ScriptWriter(Protocol):
    name: str

    def write(self, stories: list[Story]) -> Episode: ...


def credited_outlets(story: Story, banned: set[str] | tuple[str, ...] = ()) -> list[str]:
    """The outlets the show may credit: never a newsletter or a forum."""
    names = []
    for name in [*(story.outlets or []), story.source]:
        if name and not is_banned_name(name, tuple(banned)) and name not in names:
            names.append(name)
    return names


def _story_block(stories: list[Story], banned: set[str] | tuple[str, ...] = ()) -> str:
    """The story material the writer and the critic both see. A researched story with at least two
    checked quotes shows them instead of the article text; with fewer, it shows both. Newsletters
    and forums are left out of "Covered by", so the script can't credit them."""
    out = []
    for i, s in enumerate(stories, 1):
        part = f"STORY {i}: {s.headline or s.title}\n"
        outlets = credited_outlets(s, banned)
        if outlets:
            part += f"Covered by: {', '.join(outlets)}\n"
        if s.first_reported:
            part += f"First reported: {s.first_reported}\n"
        part += f"Key fact: {s.key_fact}\nSummary: {clean_text(s.summary)}\n"
        if s.evidence:
            part += "Evidence (verbatim quotes from the sources):\n" + "".join(
                f'[E{k}] "{e.quote}" ({(urlsplit(e.url).hostname or "source").removeprefix("www.")})\n'
                for k, e in enumerate(s.evidence, 1))
        if s.body and len(s.evidence) < 2:
            part += f"Article text:\n{clean_text(s.body)}\n"
        out.append(part)
    return "\n".join(out)


def description_footer(stories: list[Story]) -> str:
    """The Sources list: each story's headline and link, never a newsletter's or forum's."""
    lines = []
    for s in stories:
        name = s.headline or s.title
        name = "" if script_problems(name) else name
        url = "" if is_aggregator_url(s.url) or is_newsletter_url(s.url) else s.url
        if name or url:
            lines.append(f"- {name}: {url}" if name and url else f"- {name or url}")
    return "\n\nSources:\n" + "\n".join(lines) + "\n\nMade with AI. #AI #AINews #Shorts"


def default_description(stories: list[Story]) -> str:
    return f"Today's top {len(stories)} AI stories in about two minutes."


def _story_segment(story: Story, text: str, headline: str = "", key_fact: str = "") -> Segment:
    return Segment(kind="story", text=text.strip(), headline=(headline or story.headline or story.title).strip(),
                   key_fact=(key_fact or story.key_fact).strip(), source=story.source, url=story.url)


def _first_words(text: str, limit: int) -> str:
    """At most ``limit`` words, cut at a sentence end when one is reasonably close."""
    clipped = " ".join(text.split()[:limit])
    if clipped.endswith((".", "!", "?")):
        return clipped
    end = max(clipped.rfind(". "), clipped.rfind("! "), clipped.rfind("? "))
    if end > len(clipped) // 2:
        return clipped[:end + 1]
    return clipped.rstrip(",;:") + "."


def template_segment(story: Story) -> Segment:
    """The headline, then the start of the cleaned summary; the headline once, even when the summary
    opens by restating it."""
    headline = (story.headline or story.title).rstrip(".").strip()
    summary = clean_text(story.summary)
    sentences = [x for x in summary.replace("! ", "!\n").replace("? ", "?\n").replace(". ", ".\n").split("\n") if x]
    if sentences and repetition(f"{headline}. {sentences[0]}", headline):
        sentences = sentences[1:]
    if sentences and SequenceMatcher(None, headline.lower(), sentences[0].lower().rstrip(".")).ratio() >= 0.6:
        sentences = sentences[1:]  # the summary starts by restating the headline
    head = _event_words(headline)
    if sentences and head and len(head & _event_words(sentences[0])) / len(head) >= 0.75:
        # ... in other words, with more in it ("OpenAI launches GPT-6" / "OpenAI has launched GPT-6, its
        # largest model"): the summary alone says it once; the headline is still on screen.
        return _story_segment(story, _first_words(" ".join(sentences), STORY_WORDS[1]))
    body = " ".join(sentences)
    body = _first_words(body, STORY_WORDS[1] - len(headline.split())) if body else ""
    return _story_segment(story, f"{headline}. {body}".strip())


def assemble(data: dict, stories: list[Story], *, strict: bool = True, outro: str | None = None) -> Episode:
    """Writer JSON -> Episode. With ``strict=False`` a missing segment gets template text instead of failing.
    ``outro`` is the show's fixed outro, used instead of any the model wrote."""
    items = data.get("segments", [])
    if strict and len(items) != len(stories):
        raise ValueError(f"writer returned {len(items)} segments for {len(stories)} stories")
    if len(items) != len(stories):
        log.warning("writer returned %d segments for %d stories; filling the gaps from templates",
                    len(items), len(stories))
    segments = [Segment(kind="intro", text=(data.get("intro") or "").strip())]
    for i, story in enumerate(stories):
        item = items[i] if i < len(items) else None
        if item and (item.get("text") or "").strip():
            segments.append(_story_segment(story, item["text"], item.get("headline", ""), item.get("key_fact", "")))
        else:
            segments.append(template_segment(story))
    segments.append(Segment(kind="outro", text=(outro if outro is not None else data.get("outro") or "").strip()))
    if strict and not (segments[0].text and segments[-1].text):
        raise ValueError("writer left out the intro or outro")
    return Episode(
        title=(data.get("title") or "").strip()[:95],
        description=(data.get("description") or "").strip() + description_footer(stories),
        tags=[t for t in data.get("tags", []) if isinstance(t, str)][:15],
        segments=segments,
        stories=stories,
    )


class LLMWriter:
    name = "llm"

    def __init__(self, llm: LLM, show: str, host: str, *, outro: str = DEFAULT_OUTRO,
                 banned: set[str] | tuple[str, ...] = (), recent_intros: list[str] | tuple[str, ...] = ()):
        self.llm, self.show, self.host = llm, show, host
        self.outro, self.banned, self.recent_intros = outro, set(banned), list(recent_intros)

    def system(self, n: int) -> str:
        return WRITER_SYSTEM.format(persona=load_persona(self.show, self.host), n=n,
                                    lo=STORY_WORDS[0], hi=STORY_WORDS[1])

    def draft(self, stories: list[Story]) -> dict:
        recent = "".join(f"- {t}\n" for t in self.recent_intros)
        user = (f"Today is {date.today():%A, %B %d, %Y}.\n\n"
                + (f"Recent intros (open differently from all of these):\n{recent}\n" if recent else "")
                + _story_block(stories, self.banned))
        return self.llm.json(self.system(len(stories)), user, stage="writer", schema=SCRIPT_SCHEMA)

    def write(self, stories: list[Story]) -> Episode:
        episode = assemble(self.draft(stories), stories, outro=self.outro)
        for seg in episode.story_segments:
            words = len(seg.text.split())
            if not STORY_WORDS[0] - 8 <= words <= STORY_WORDS[1] + 10:
                log.warning("segment %r is %d words (target %d-%d)", seg.headline, words, *STORY_WORDS)
        return episode


class TemplateWriter:
    """No-key fallback: headline plus the first sentences of each summary. Also the last resort for a
    single intro or story the writer agent couldn't get right."""

    name = "template"

    def __init__(self, show: str, host: str, *, outro: str = DEFAULT_OUTRO,
                 recent_intros: list[str] | tuple[str, ...] = ()):
        self.show, self.host, self.outro_text = show, host, outro
        self.recent_intros = list(recent_intros)

    def intro(self, n: int) -> Segment:
        """Today's hook from the rotation, skipping any a recent episode opened with."""
        recent = {opening(t) for t in self.recent_intros}
        start = date.today().toordinal()
        hooks = [TEMPLATE_HOOKS[(start + k) % len(TEMPLATE_HOOKS)].format(host=self.host, show=self.show)
                 for k in range(len(TEMPLATE_HOOKS))]
        hook = next((h for h in hooks if opening(h) not in recent), hooks[0])
        return Segment(kind="intro", text=f"{hook} Here are the AI stories you need today.")

    def outro(self) -> Segment:
        return Segment(kind="outro", text=self.outro_text)

    def write(self, stories: list[Story]) -> Episode:
        segments = [self.intro(len(stories)), *(template_segment(s) for s in stories), self.outro()]
        title = f"AI News Today: {stories[0].headline or stories[0].title}"[:95]
        return Episode(
            title="AI News Today" if script_problems(title) else title,
            description=f"Today's top {len(stories)} AI stories in two minutes." + description_footer(stories),
            tags=["AI", "AI news", "artificial intelligence", "tech news", "shorts"],
            segments=segments,
            stories=stories,
        )


def _script_rows(episode: Episode) -> str:
    rows = {"intro": episode.segments[0].text, "outro": episode.segments[-1].text,
            "segments": [{"story": i, "headline": s.headline, "key_fact": s.key_fact, "text": s.text,
                          "words": len(s.text.split())}
                         for i, s in enumerate(episode.story_segments, 1)]}
    return json.dumps(rows, indent=1)


class CriticWriter:
    """Writer agent: draft, check in code and with a critic, rewrite only what fails."""

    name = "agent"

    def __init__(self, llm: LLM, critic: LLM | None, show: str, host: str, max_repairs: int = 2, *,
                 outro: str = DEFAULT_OUTRO, banned: set[str] | tuple[str, ...] = (),
                 recent_intros: list[str] | tuple[str, ...] = ()):
        # Without a critic (SHORTS_AGENTS=off) it is the one-call writer plus the code checks and fallbacks.
        self.name = "agent" if critic is not None else "llm"
        self.base = LLMWriter(llm, show, host, outro=outro, banned=banned, recent_intros=recent_intros)
        self.llm, self.critic, self.max_repairs = llm, critic, max_repairs
        self.template = TemplateWriter(show, host, outro=outro, recent_intros=recent_intros)
        # What each round found and what fell back to templates; saved as 03-review.json.
        self.report: dict = {"rounds": [], "fallbacks": [], "dropped": [], "critic_errors": 0}

    def lint(self, episode: Episode, stories: list[Story]) -> list[Issue]:
        return lint_episode(episode, stories, self.frame(), banned=self.base.banned,
                            recent_intros=self.base.recent_intros, description=_own_description(episode))

    def write(self, stories: list[Story]) -> Episode:
        self.report = {"rounds": [], "fallbacks": [], "dropped": [], "critic_errors": 0}
        draft = self.base.draft(stories)
        episode = assemble(draft, stories, strict=False, outro=self.base.outro)
        items = draft.get("segments") or []
        for i in range(len(stories)):
            if i >= len(items) or not (items[i].get("text") or "").strip():
                self.report["fallbacks"].append({"part": f"story {i + 1}", "why": "the writer left it out"})
        critic: list[Issue] = []
        for round_no in range(self.max_repairs + 1):
            issues = self.lint(episode, stories)
            rules = len(fatal(issues))
            critic = []
            try:
                critic = self.review(episode, stories) if self.critic is not None else []
                issues += critic
            except BudgetExceeded as exc:
                log.warning("      %s", exc)
                self.report["critic_errors"] += 1
            except Exception as exc:  # the critic is an extra check, never a reason to fail
                log.warning("      critic failed (%s); using the rule checks only", exc)
                self.report["critic_errors"] += 1
            # A rewrite can't change the title or description; settle() replaces a bad one.
            todo = [i for i in fatal(issues) if i.code not in SETTLE_ONLY]
            self.report["rounds"].append({"fatal": rules, "critic": len(fatal(critic)),
                                          "problems": [i.line()[2:] for i in fatal(issues)]})
            if not todo or round_no == self.max_repairs:
                break
            log.info("      writer repair round %d: %s", round_no + 1, "; ".join(i.line()[2:] for i in todo))
            try:
                episode = self.revise(episode, stories, todo, stage=f"writer-repair-{round_no + 1}")
            except BudgetExceeded as exc:
                log.warning("      %s", exc)
                break
            except Exception as exc:
                log.warning("      rewrite failed (%s)", exc)
                break
        return self.settle(episode, stories, fatal(self.lint(episode, stories)) + fatal(critic))

    def frame(self) -> str:
        """What the intro and outro may mention besides the stories."""
        return f"{self.base.show} {self.base.host} {date.today():%A, %B %d, %Y}"

    def review(self, episode: Episode, stories: list[Story]) -> list[Issue]:
        user = (f"The intro and outro may name the show ({self.base.show}), the host ({self.base.host}) and today's "
                f"date ({date.today():%A, %B %d, %Y}); those are not claims to check.\n\n"
                f"=== SCRIPT ===\n{_script_rows(episode)}\n\n=== STORY MATERIAL ===\n"
                f"{_story_block(stories, self.base.banned)}")
        system = CRITIC_SYSTEM + (CRITIC_EVIDENCE_NOTE if any(s.evidence for s in stories) else "")
        data = self.critic.json(system, user, stage="critic", schema=REVIEW_SCHEMA)
        issues = []
        claims = [c for c in data.get("intro", []) if c]  # the outro is the show's own, so not checked
        if claims:
            issues.append(Issue("intro_problem", "the intro makes claims the material does not support: "
                                                 + "; ".join(claims)))
        for item in data.get("segments", []):
            idx, claims = item.get("story", 0) - 1, [c for c in item.get("unsupported", []) if c]
            if 0 <= idx < len(stories) and claims:
                issues.append(Issue("unsupported_claim", "not supported by the material: " + "; ".join(claims), idx))
            quality = [q for q in item.get("quality", []) if q]
            if 0 <= idx < len(stories) and quality:
                issues.append(Issue("quality", "; ".join(quality), idx))
        return issues

    def revise(self, episode: Episode, stories: list[Story], problems: list[Issue], *, stage: str) -> Episode:
        user = REVISE_PROMPT.format(problems="\n".join(p.line() for p in problems), script=_script_rows(episode),
                                    material=_story_block(stories, self.base.banned))
        data = self.llm.json(self.base.system(len(stories)), user, stage=stage, schema=REVISION_SCHEMA)
        whole_script = any(p.index is None and p.code not in FRAME_CODES for p in problems)
        allowed = set(range(len(stories))) if whole_script else {p.index for p in problems}
        segments = list(episode.segments)
        for item in data.get("segments", []):
            idx = item.get("story", 0) - 1
            if idx in allowed and 0 <= idx < len(stories) and (item.get("text") or "").strip():
                segments[idx + 1] = _story_segment(stories[idx], item["text"], item.get("headline", ""),
                                                   item.get("key_fact", ""))
        if (data.get("intro") or "").strip():
            segments[0] = Segment(kind="intro", text=data["intro"].strip())
        # the outro is the show's own, never rewritten
        return replace(episode, segments=segments)

    def settle(self, episode: Episode, stories: list[Story], remaining: list[Issue]) -> Episode:
        """Last resort for problems the rewrites didn't fix: template text or a trim, segment by segment."""
        segments = list(episode.segments)
        replaced: set[int] = set()  # positions already swapped for template text
        for issue in remaining:
            pos = _segment_position(issue)
            if pos is not None and pos in replaced:
                continue
            if issue.code != "too_long" and pos is not None:
                replaced.add(pos)
            if issue.code in ("missing_intro", "intro_problem"):
                log.warning("      the intro still has a problem (%s); using the standard intro", issue.detail)
                segments[0] = self.template.intro(len(stories))
                self.report["fallbacks"].append({"part": "intro", "why": issue.detail})
            elif issue.code in ("missing_outro", "outro_problem"):
                log.warning("      the outro still has a problem (%s); using the standard outro", issue.detail)
                segments[-1] = self.template.outro()
                self.report["fallbacks"].append({"part": "outro", "why": issue.detail})
            elif issue.index is None:
                continue
            elif issue.code == "too_long":
                seg = segments[issue.index + 1]
                segments[issue.index + 1] = replace(seg, text=_first_words(seg.text, STORY_WORDS[1]))
            else:
                log.warning("      story %d still has a problem (%s); using its summary instead",
                            issue.index + 1, issue.detail)
                segments[issue.index + 1] = template_segment(stories[issue.index])
                self.report["fallbacks"].append({"part": f"story {issue.index + 1}", "why": issue.detail})
        episode = replace(episode, segments=segments)
        episode = self.drop_unfit(episode, stories)
        stories = episode.stories
        if not stories:
            return episode
        title_why = ("missing" if not episode.title else
                     "; ".join(script_problems(episode.title, self.base.banned)
                               + [f"uses {n}, which is not in the material"
                                  for n in unsupported_numbers(episode.title, episode_material(stories, self.frame()))]))
        if title_why:
            episode.title = self.template.write(stories).title
            if script_problems(episode.title, self.base.banned):
                episode.title = "AI News Today"
            self.report["fallbacks"].append({"part": "title", "why": title_why})
        desc_why = "; ".join(script_problems(_own_description(episode), self.base.banned))
        if desc_why:
            episode.description = default_description(stories) + description_footer(stories)
            self.report["fallbacks"].append({"part": "description", "why": desc_why})
        return trim_to_fit(episode, TARGET_MAX_SECONDS)

    def drop_unfit(self, episode: Episode, stories: list[Story]) -> Episode:
        """Leave out a story whose segment, even as template text, would read out source talk, repeat
        itself or say nothing beyond its headline. Better one story fewer than a bad one."""
        issues = lint_episode(episode, stories, self.frame(), banned=self.base.banned)
        bad = sorted({i.index for i in issues
                      if i.index is not None and i.code in ("source_talk", "repeats", "says_little", "copied")})
        if not bad:
            return replace(episode, stories=list(stories))
        for idx in bad:
            log.warning("      leaving out story %d (%s): not fit to air", idx + 1, stories[idx].headline)
            self.report["dropped"].append({"story": idx + 1, "headline": stories[idx].headline or stories[idx].title})
        keep = [i for i in range(len(stories)) if i not in bad]
        kept_stories = [stories[i] for i in keep]
        segments = [episode.segments[0], *(episode.segments[i + 1] for i in keep), episode.segments[-1]]
        gone = [_event_words(stories[i].headline or stories[i].title) | _event_words(episode.segments[i + 1].headline)
                for i in bad]

        def teases(text: str) -> bool:
            said = _event_words(text)
            return any(len(words & said) >= 2 for words in gone)

        title, description = episode.title, _own_description(episode)
        if teases(episode.segments[0].text):
            segments[0] = self.template.intro(len(kept_stories))  # the intro teased a story that's gone
            self.report["fallbacks"].append({"part": "intro", "why": "teased a story that was left out"})
        if kept_stories and teases(title):
            title = self.template.write(kept_stories).title
            self.report["fallbacks"].append({"part": "title", "why": "named a story that was left out"})
        if kept_stories and teases(description):
            description = default_description(kept_stories)
            self.report["fallbacks"].append({"part": "description", "why": "named a story that was left out"})
        return replace(episode, segments=segments, stories=kept_stories, title=title,
                       description=description + description_footer(kept_stories))

    def shorten(self, episode: Episode, stories: list[Story], seconds_over: float) -> Episode:
        """A shorter rewrite, checked like the draft: any rewritten part that now fails keeps its old text."""
        if self.critic is None:
            return episode  # no rewrites without the agents; trim_to_fit cuts in code
        words = int(seconds_over * WORDS_PER_SECOND) + 4
        problem = Issue("too_long_total", f"the voiced episode runs {seconds_over:.0f}s too long; cut about {words} "
                                          "words, mostly from the longest segments")
        try:
            shorter = self.revise(episode, stories, [problem], stage="writer-trim")
        except Exception as exc:
            log.warning("      trim rewrite failed (%s); trimming in code", exc)
            return episode
        old = episode.segments
        # A trim changes only the stories' spoken text; the intro, outro, headlines and key facts
        # stay as checked.
        segments = [replace(new, headline=prev.headline, key_fact=prev.key_fact) if new.kind == "story" else prev
                    for new, prev in zip(shorter.segments, old)]
        shorter = replace(shorter, segments=segments)
        issues = fatal(self.lint(shorter, stories))
        try:
            issues += self.review(shorter, stories)
        except Exception as exc:
            log.warning("      critic failed on the trim (%s); using the rule checks only", exc)
        for issue in issues:
            pos = _segment_position(issue)
            if pos is not None and segments[pos] != old[pos]:
                log.warning("      the trim broke story %d (%s); keeping the old text", pos, issue.detail)
                segments[pos] = old[pos]
        shorter = replace(shorter, segments=segments)
        return shorter if predicted_seconds(shorter) < predicted_seconds(episode) else episode


def _own_description(episode: Episode) -> str:
    """The writer's description, without the sources footer."""
    return episode.description.split("\n\nSources:\n")[0]


def _segment_position(issue: Issue) -> int | None:
    """Where in ``episode.segments`` an issue points: 0 intro, -1 outro, i + 1 story i, None the whole episode."""
    if issue.code in ("missing_intro", "intro_problem"):
        return 0
    if issue.code in ("missing_outro", "outro_problem"):
        return -1
    return None if issue.index is None else issue.index + 1


def trim_to_fit(episode: Episode, max_seconds: float) -> Episode:
    """Cut the longest story segments at sentence ends until the predicted length fits."""
    segments = list(episode.segments)
    for _ in range(40):
        ep = replace(episode, segments=segments)
        if predicted_seconds(ep) <= max_seconds:
            return ep
        idx = max(range(1, len(segments) - 1), key=lambda i: len(segments[i].text.split()), default=None)
        if idx is None:
            break
        words = len(segments[idx].text.split())
        if words <= STORY_WORDS[0] - 8:
            break
        segments[idx] = replace(segments[idx], text=_first_words(segments[idx].text, words - 4))
    return replace(episode, segments=segments)


def shorten(writer: ScriptWriter, episode: Episode, stories: list[Story], measured_seconds: float,
            max_seconds: float = TARGET_MAX_SECONDS) -> Episode:
    """The voiced episode came out too long: let the writer agent cut it, then make sure in code."""
    over = measured_seconds - max_seconds
    # The voice runs at its own pace, so scale the target by how far off the prediction was
    # for the script that was actually voiced.
    pace = measured_seconds / max(predicted_seconds(episode), 1.0)
    if isinstance(writer, CriticWriter):
        episode = writer.shorten(episode, stories, over)
    return trim_to_fit(episode, max_seconds / max(pace, 1.0) - 2)


def build_writer(llm: LLM | None, show: str, host: str, *, agents: bool = False, critic: LLM | None = None,
                 max_repairs: int = 2, outro: str = DEFAULT_OUTRO, banned: set[str] | tuple[str, ...] = (),
                 recent_intros: list[str] | tuple[str, ...] = ()) -> ScriptWriter:
    if not llm:
        return TemplateWriter(show, host, outro=outro, recent_intros=recent_intros)
    if agents:
        return CriticWriter(llm, critic or llm, show, host, max_repairs, outro=outro, banned=banned,
                            recent_intros=recent_intros)
    # One writer call, no critic and no rewrites, but the same code checks and fallbacks.
    return CriticWriter(llm, None, show, host, 0, outro=outro, banned=banned, recent_intros=recent_intros)


class WriterFailed(RuntimeError):
    """The AI writer failed and a template-read script isn't allowed on air."""


def write_episode(writer: ScriptWriter, stories: list[Story], show: str, host: str, *,
                  allow_template: bool = True, outro: str = DEFAULT_OUTRO) -> Episode:
    try:
        episode = writer.write(stories)
    except Exception as exc:
        if isinstance(writer, TemplateWriter):
            raise
        if not allow_template:
            raise WriterFailed(f"the {writer.name} writer failed ({exc}); not publishing a template-read "
                               "script instead") from exc
        log.warning("%s writer failed (%s); falling back to template", writer.name, exc)
        recent = getattr(writer, "recent_intros", None) or getattr(getattr(writer, "base", None), "recent_intros", ())
        return TemplateWriter(show, host, outro=outro, recent_intros=recent).write(stories)
    report = getattr(writer, "report", {})
    templated = {f["part"] for f in report.get("fallbacks", []) if str(f.get("part", "")).startswith("story")}
    templated -= {f"story {d.get('story')}" for d in report.get("dropped", [])}  # left out, so not aired
    aired = len(episode.story_segments)
    if not allow_template and aired and len(templated) * 2 > aired:
        raise WriterFailed(f"the {writer.name} writer got only {aired - len(templated)} of {aired} stories right; "
                           "not publishing a script that is mostly summaries read out")
    return episode


def episode_json(episode: Episode) -> str:
    from dataclasses import asdict

    return json.dumps(asdict(episode), indent=2, default=str)


def episode_from_json(text: str) -> Episode:
    """Back from ``episode_json``: enough to voice or re-render a script (the stories are left out)."""
    data = json.loads(text)
    segments = [Segment(**{k: v for k, v in seg.items() if k in Segment.__dataclass_fields__})
                for seg in data.get("segments", [])]
    return Episode(title=data.get("title", ""), description=data.get("description", ""),
                   tags=list(data.get("tags", [])), segments=segments)


class IntroLog:
    """The last episodes' intros, so today's hook can be told to open differently."""

    KEEP = 7

    def __init__(self, path: Path):
        self.path = path
        try:
            entries = json.loads(path.read_text()) if path.exists() else []
        except (OSError, ValueError):
            entries = []
        self.entries: list[dict] = [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else []

    @property
    def recent(self) -> list[str]:
        return [e["text"] for e in self.entries if isinstance(e, dict) and e.get("text")]

    def add(self, text: str) -> None:
        self.entries = [*self.entries, {"date": date.today().isoformat(), "text": text}][-self.KEEP:]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=1))
