import json
import threading
import time
from collections import Counter
from dataclasses import replace
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest
import requests

from shorts import research, web
from shorts.checks import norm_url
from shorts.config import Config
from shorts.llm import OpenAILLM, Usage
from shorts.models import Story
from shorts.research import (RESEARCH_SCHEMA, AgentResearcher, NoResearch, TavilyResearcher, build_researcher,
                             swap_failing)
from shorts.selection import SeenStore, settle
from shorts.web import JINA_READER, Doc, Jina, Tavily, TavilyCredits, Web
from tests.test_agents import ScriptedLLM, _openai, _resp
from tests.test_agents import _story as _feed_story

TODAY = date(2026, 9, 26)
PUBLISHED = datetime(2026, 9, 26, 6, tzinfo=timezone.utc)
HEADLINE = "Nvidia unveils Rubin GPU at GTC Paris"
URL = "https://lab.ai/rubin"
BLOG = "https://blogs.nvidia.com/rubin"
HIT = "https://nvidianews.nvidia.com/news/rubin"
Q1 = "Nvidia said the Rubin GPU will ship to cloud providers early next year."
Q2 = "Jensen Huang showed the Rubin GPU on stage at GTC Paris on Thursday."
Q3 = "Each Rubin package pairs two dies with 288 gigabytes of HBM4 memory, Nvidia said."
OFF_TOPIC = "Apple released a new camera app for the iPhone with manual focus controls."
INVENTED = "Nvidia said Rubin will be free for every developer on day one."
FILLER = "The keynote also covered partner programs, developer tools and a long list of customer stories."
THIN = "Nvidia Rubin GPU. Subscribe to continue reading."


def _article(*sentences):
    text = " ".join(sentences)
    while len(text) < 700:
        text += " " + FILLER
    return text


ARTICLE = _article(Q1, Q2)


def _pick(headline=HEADLINE, url=URL, body=""):
    return Story(title=headline, url=url, source="TLDR AI", published=PUBLISHED, summary=f"{headline}, says TLDR.",
                 body=body, headline=headline, outlets=["TLDR AI"])


def _answer(quotes=(), url="", matches="yes", first="", source=""):
    return json.dumps({"matches_pick": matches, "first_reported": first, "source_url": source,
                       "evidence": [{"quote": q, "url": url} for q in quotes], "note": ""})


def _pool(*docs):
    return {norm_url(d.url): d for d in docs}


def _jina_page(url, title, text, published=""):
    return f"Title: {title}\nURL Source: {url}\nPublished Time: {published}\n\nMarkdown Content:\n{text}"


class _Reply:
    def __init__(self, status=200, data=None, text="", url=""):
        self.status_code, self._data, self.text, self.url = status, data, text, url

    def json(self):
        if self._data is None:
            raise ValueError("not json")
        return self._data

    def close(self):
        pass


class FakeHTTP:
    """Tavily search and extract, Jina Reader and HEAD redirects served from dicts; records every request."""

    def __init__(self):
        self.extract, self.search, self.jina, self.redirects = {}, [], {}, {}
        self.posts, self.gets, self.heads = [], [], []
        self._lock = threading.Lock()

    def post(self, url, json=None, headers=None, timeout=None):
        with self._lock:
            self.posts.append((url.rsplit("/", 1)[-1], json))
        if url.endswith("/search"):
            return _Reply(data={"results": self.search})
        urls = json["urls"]
        return _Reply(data={"results": [{"url": u, "raw_content": self.extract[u]} for u in urls if u in self.extract],
                            "failed_results": [{"url": u} for u in urls if u not in self.extract]})

    def get(self, url, headers=None, timeout=None, **kwargs):
        with self._lock:
            self.gets.append(url)
        page = self.jina.get(url.removeprefix(JINA_READER))
        return _Reply(text=page) if page else _Reply(status=404, text="not found")

    def head(self, url, **kwargs):
        with self._lock:
            self.heads.append(url)
        return _Reply(url=self.redirects.get(url, url))


@pytest.fixture
def http(monkeypatch):
    fake = FakeHTTP()
    namespace = SimpleNamespace(get=fake.get, post=fake.post, head=fake.head, RequestException=requests.RequestException)
    monkeypatch.setattr(web, "requests", namespace)
    monkeypatch.setattr(research, "requests", namespace)
    return fake


def _web(key="tvly", credits=None):
    tavily = Tavily(key, credits or TavilyCredits(None)) if key else None
    return Web(tavily=tavily, jina=Jina(), coverage=None)


def _researcher(seen=()):
    return AgentResearcher(None, _web(""), [], 30, set(seen), today=TODAY)


class RoutingResponses:
    """Thread-safe fake Responses API: ``route(format name, kwargs)`` gives each reply, outside the lock."""

    def __init__(self, route):
        self.route, self.calls = route, []
        self._lock = threading.Lock()

    def create(self, **kwargs):
        with self._lock:
            self.calls.append(kwargs)
        reply = self.route(kwargs["text"]["format"]["name"], kwargs)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _routed(route, model="gpt-5", usage=None):
    fake = RoutingResponses(route)
    return OpenAILLM("", model, usage or Usage(), client=SimpleNamespace(responses=fake)), fake


def _index(name):
    return int(name.rsplit("_", 1)[1]) - 1


PAIRS = [("Nvidia", "Rubin"), ("OpenAI", "Atlas"), ("Anthropic", "Claude"), ("Google", "Gemini"),
         ("Meta", "Llama"), ("Apple", "Siri"), ("Amazon", "Nova"), ("Microsoft", "Copilot")]


def _eight():
    stories, quotes = [], []
    for co, product in PAIRS:
        q = [f"{co} said {product} can plan and book a whole trip without help.",
             f"{product} reaches {co} customers in forty countries starting this week."]
        stories.append(_pick(f"{co} unveils {product} agent", f"https://{co.lower()}.example/{product.lower()}",
                             body=_article(*q)))
        quotes.append(q)
    return stories, quotes


def _join_researchers():
    for t in threading.enumerate():
        if t.name.startswith("research"):
            t.join(10)


# --- the checks in code ----------------------------------------------------------------------

def test_invented_quotes_are_dropped_and_counted_and_moved_quotes_follow_their_page():
    a = Doc(url=URL, title="Nvidia Rubin", text=_article(Q1, Q2), via="tavily")
    b = Doc(url=BLOG, title="Rubin specs", text=_article(Q3), via="jina")
    data = {"matches_pick": "yes", "evidence": [{"quote": Q1, "url": URL}, {"quote": Q3, "url": URL},
                                                 {"quote": INVENTED, "url": URL}, {"quote": Q1, "url": URL}]}
    f = _researcher().verify(_pick(body=a.text), data, _pool(a, b))
    assert [(e.quote, e.url) for e in f.evidence] == [(Q1, URL), (Q3, BLOG)]
    assert f.dropped == 1 and f.status == "verified"
    made_up = [INVENTED, "Rubin GPUs cost ten dollars an hour, Nvidia said."]
    f = _researcher().verify(_pick(body=a.text), {"evidence": [{"quote": q, "url": URL} for q in made_up]}, _pool(a, b))
    assert f.evidence == [] and f.dropped == 2 and f.status == "thin"


def test_quotes_from_an_unrelated_page_are_dropped():
    a = Doc(url=URL, title="Nvidia Rubin", text=_article(Q1, Q2), via="tavily")
    other = Doc(url="https://apple.example/camera", title="New iPhone camera app", text=_article(OFF_TOPIC), via="search")
    data = {"evidence": [{"quote": OFF_TOPIC, "url": other.url}, {"quote": Q1, "url": URL},
                         {"quote": OFF_TOPIC.replace("Apple", "apple"), "url": URL}]}
    f = _researcher().verify(_pick(body=a.text), data, _pool(a, other))
    assert [e.quote for e in f.evidence] == [Q1] and f.status == "thin"


def test_at_most_six_quotes_of_at_most_300_characters():
    long_quotes = [f"Point {k}: Nvidia said the Rubin GPU " + " ".join(["delivers much faster training"] * 14)
                   + f" in test {k}." for k in range(8)]
    doc = Doc(url=URL, title="Nvidia Rubin", text=" ".join(long_quotes), via="tavily")
    data = {"evidence": [{"quote": q, "url": URL} for q in long_quotes]}
    f = _researcher().verify(_pick(body=doc.text), data, _pool(doc))
    assert len(f.evidence) == 6 and f.dropped == 0
    for e, original in zip(f.evidence, long_quotes):
        assert len(e.quote) <= 300 and len(e.quote) > 250 and original.startswith(e.quote)


def _first(value, *docs):
    return _researcher().verify(_pick(body=ARTICLE), {"first_reported": value}, _pool(*docs)).first_reported


def test_dates_count_only_when_a_page_backs_them():
    dated = Doc(url=URL, text=_article(Q1, Q2), published="2026-09-25", via="tavily")
    assert _first("2026-09-25", dated) == "2026-09-25"
    assert _first("2026-09-24", dated) == "2026-09-24"  # a day off the page's date
    assert _first("2026-09-26", dated) == "2026-09-26"
    assert _first("2026-09-23", dated) == ""
    for line in ("Nvidia first showed Rubin at a press event on Sept. 22.",
                 "Nvidia first showed Rubin at a press event on September 22nd.",
                 "Nvidia first showed Rubin at a press event on 22 September.",
                 "Posted 2026-09-22 by the Nvidia newsroom."):
        page = Doc(url=BLOG, text=_article(Q3, line), via="jina")
        assert _first("2026-09-22", dated, page) == "2026-09-22", line
        assert _first("2026-09-21", page) == ""
    assert _first("2026-09-22T10:00:00Z", Doc(url=BLOG, text=_article("Nvidia said on September 22 that it ships."))) \
        == "2026-09-22"
    assert _first("2026-09-22", Doc(url=BLOG, text=_article(Q3))) == ""
    assert _first("last Tuesday", dated) == "" and _first("", dated) == ""


def test_future_dates_are_ignored_even_when_a_page_says_them():
    future = Doc(url=URL, text=_article(Q1, "Rubin goes on sale on September 30."), published="2026-09-30")
    assert _first("2026-09-30", future) == ""
    assert _first("2026-09-27", Doc(url=URL, text=_article(Q1), published="2026-09-27")) == ""


def test_stale_only_with_a_backed_old_date():
    r = _researcher()
    quotes = [{"quote": Q1, "url": URL}, {"quote": Q2, "url": URL}]
    old = Doc(url=URL, text=_article(Q1, Q2), published="2026-09-20", via="tavily")
    fresh = Doc(url=URL, text=_article(Q1, Q2), published="2026-09-25", via="tavily")
    f = r.verify(_pick(body=ARTICLE), {"first_reported": "2026-09-20", "evidence": quotes}, _pool(old))
    assert (f.status, f.first_reported) == ("stale", "2026-09-20")
    f = r.verify(_pick(body=ARTICLE), {"first_reported": "2026-09-20", "evidence": quotes}, _pool(fresh))
    assert (f.status, f.first_reported) == ("verified", "")
    f = r.verify(_pick(body=ARTICLE), {"first_reported": "2026-09-25", "evidence": quotes}, _pool(fresh))
    assert (f.status, f.first_reported) == ("verified", "2026-09-25")
    f = r.verify(_pick(body=ARTICLE), {"first_reported": "2026-10-20", "evidence": quotes},
                 _pool(Doc(url=URL, text=ARTICLE, published="2026-10-20")))
    assert (f.status, f.first_reported) == ("verified", "")


def test_source_url_is_taken_only_when_read_and_the_original_is_weak():
    src = Doc(url=HIT, title="Nvidia unveils Rubin at GTC Paris", text=_article(Q1, Q2, Q3), via="search")
    thin = Doc(url=URL, text=THIN, via="tavily")
    good = Doc(url=URL, text=ARTICLE, via="tavily")
    r = _researcher()
    f = r.verify(_pick(body=THIN), {"source_url": "https://made.up/rubin"}, _pool(thin, src))
    assert f.url == "" and f.body == ""
    f = r.verify(_pick(body=ARTICLE), {"matches_pick": "yes", "source_url": HIT}, _pool(good, src))
    assert f.url == "" and f.body == ""
    f = r.verify(_pick(body=THIN), {"matches_pick": "yes", "source_url": HIT}, _pool(thin, src))
    assert (f.url, f.body) == (HIT, src.text)
    f = r.verify(_pick(url="", body=""), {"source_url": HIT + "/?utm_source=x"}, _pool(src))
    assert f.url == HIT
    f = r.verify(_pick(body=ARTICLE), {"matches_pick": "no", "source_url": HIT}, _pool(good, src))
    assert f.url == HIT and f.matches == "no" and f.status != "wrong_story"
    f = r.verify(_pick(body=ARTICLE), {"matches_pick": "no", "source_url": ""}, _pool(good, src))
    assert f.url == "" and f.status == "wrong_story"
    feed = Doc(url="https://tldr.tech/ai/rubin", title=HEADLINE, text=f"{HEADLINE}. {Q1}", via="feed")
    f = r.verify(_pick(body=THIN), {"source_url": feed.url}, _pool(thin, feed))
    assert f.url == ""
    off = Doc(url="https://apple.example/camera", title="iPhone camera", text=_article(OFF_TOPIC), via="search")
    f = r.verify(_pick(body=THIN), {"source_url": off.url}, _pool(thin, off))
    assert f.url == ""


def test_an_already_aired_source_is_refused():
    src = Doc(url=HIT, title="Nvidia unveils Rubin at GTC Paris", text=_article(Q1, Q2, Q3), via="search")
    r = _researcher(seen={"https://www.nvidianews.nvidia.com/news/rubin/?utm_source=tldr"})
    f = r.verify(_pick(body=ARTICLE), {"matches_pick": "no", "source_url": HIT}, _pool(Doc(url=URL, text=ARTICLE), src))
    assert f.url == "" and f.status == "wrong_story"


def test_nothing_from_a_link_about_another_story_counts():
    wrong = Doc(url=URL, title="Nvidia Rubin", text=_article(Q1, Q2), published="2026-09-25", via="tavily")
    src = Doc(url=HIT, title="Nvidia unveils Rubin at GTC Paris", text=_article(Q3), via="search")
    data = {"matches_pick": "no", "first_reported": "2026-09-25", "source_url": "",
            "evidence": [{"quote": Q1, "url": URL}, {"quote": Q2, "url": URL}]}
    f = _researcher().verify(_pick(body=wrong.text), data, _pool(wrong))
    assert (f.status, f.evidence, f.first_reported) == ("wrong_story", [], "")
    f = _researcher().verify(_pick(body=wrong.text), {**data, "source_url": HIT,
                                                      "evidence": data["evidence"] + [{"quote": Q3, "url": HIT}]},
                             _pool(wrong, src))
    assert f.url == HIT and [e.quote for e in f.evidence] == [Q3] and f.first_reported == "" and f.dropped == 2


# --- one researcher's tool loop ---------------------------------------------------------------

def test_researcher_searches_reads_a_hit_from_the_pool_and_verifies(http):
    hit_text = _article(Q1, Q2, Q3)
    http.extract = {URL: THIN}
    http.search = [{"url": HIT, "title": "Nvidia unveils Rubin at GTC Paris", "content": "Nvidia unveiled Rubin.",
                    "raw_content": hit_text, "published_date": "2026-09-25T09:00:00Z"}]
    llm, fake = _openai(_resp(calls=[("web_search", {"query": "Nvidia Rubin GPU GTC Paris"})], rid="r1"),
                        _resp(calls=[("read_article", {"url": HIT})], rid="r2"),
                        _resp(text=_answer([Q1, Q2], HIT, first="2026-09-25", source=HIT), rid="r3"))
    credits = TavilyCredits(None)
    r = AgentResearcher(llm, _web("tvly", credits), [], 30, set(), today=TODAY)
    story = _pick()
    r.enrich([story])
    assert story.checked == "verified" and story.first_reported == "2026-09-25"
    assert [(e.quote, e.url) for e in story.evidence] == [(Q1, HIT), (Q2, HIT)]
    assert (story.url, story.body) == (HIT, hit_text)
    assert [kind for kind, _ in http.posts] == ["extract", "search"] and credits.run_used == 2
    assert http.posts[1][1]["query"] == "Nvidia Rubin GPU GTC Paris"
    assert http.gets == [JINA_READER + URL]  # the prefetch tried Jina once; the hit came from the pool
    first = fake.calls[0]
    assert first["text"]["format"]["name"] == "research_1" and first["text"]["format"]["schema"] == RESEARCH_SCHEMA
    assert {t["name"] for t in first["tools"]} == {"read_article", "web_search"}
    assert f"ARTICLE ({URL}, via tavily" in first["input"] and THIN in first["input"]
    search_out = fake.calls[1]["input"][0]["output"]
    read_out = fake.calls[2]["input"][0]["output"]
    assert search_out.startswith(f"1. Nvidia unveils Rubin at GTC Paris | {HIT} | 2026-09-25")
    assert read_out.startswith(f"URL: {HIT}\n") and "Published: 2026-09-25" in read_out and Q3 in read_out
    for out in (search_out, read_out):
        with pytest.raises(ValueError):
            json.loads(out)
    assert [c.get("previous_response_id") for c in fake.calls] == [None, "r1", "r2"]
    row = r.rows[0]
    assert row["tools"] == ["web_search", "read_article"] and row["status"] == "verified"
    assert row["dropped_quotes"] == 0 and row["quotes"] == [Q1, Q2]


def test_read_article_refuses_unknown_links_with_no_http(http):
    http.jina = {BLOG: _jina_page(BLOG, "Rubin specs", _article(Q3), "2026-09-25T08:00:00Z")}
    credits = TavilyCredits(None)
    llm, fake = _openai(_resp(calls=[("read_article", {"url": "https://evil.example/rubin"}),
                                     ("read_article", {"url": BLOG})]),
                        _resp(text=_answer([Q1, Q3])))
    r = AgentResearcher(llm, _web("tvly", credits), [], 30, set(), today=TODAY)
    f = r.investigate(0, _pick(body=_article(Q1, Q2, f"Nvidia's post is at {BLOG}.")), time.monotonic() + 100)
    unknown, linked = [o["output"] for o in fake.calls[1]["input"]]
    assert unknown.startswith("error: unknown link")
    assert linked.startswith(f"URL: {BLOG}\n")  # a link inside the article is allowed
    assert http.gets == [JINA_READER + BLOG] and http.posts == [] and credits.run_used == 0
    assert [e.url for e in f.evidence] == [URL, BLOG] and f.status == "verified"


def test_read_article_pays_for_one_tavily_extract_at_most(http):
    first, second = "https://lab.ai/first", "https://lab.ai/second"
    http.extract = {first: _article(Q3), second: _article(Q3)}
    llm, fake = _openai(_resp(calls=[("read_article", {"url": first}), ("read_article", {"url": second})]),
                        _resp(text=_answer([Q1, Q2], URL)))
    r = AgentResearcher(llm, _web("tvly"), [], 30, set(), today=TODAY)
    r.investigate(0, _pick(body=_article(Q1, Q2, f"See {first} and {second} for more.")), time.monotonic() + 100)
    paid, unpaid = [o["output"] for o in fake.calls[1]["input"]]
    assert paid.startswith(f"URL: {first}") and unpaid == "error: could not read that page"
    assert [kind for kind, _ in http.posts] == ["extract"] and len(http.gets) == 2


def test_third_search_and_fifth_tool_call_get_errors(http):
    http.search = [{"url": HIT, "title": "Nvidia Rubin", "content": "Rubin.", "raw_content": _article(Q3)}]
    search, read = ("web_search", {"query": "Nvidia Rubin"}), ("read_article", {"url": URL})
    llm, fake = _openai(_resp(calls=[search, search, search], rid="r1"), _resp(calls=[read, read], rid="r2"),
                        _resp(text=_answer([Q1, Q2], URL), rid="r3"))
    r = AgentResearcher(llm, _web("tvly"), [], 30, set(), today=TODAY)
    f = r.investigate(0, _pick(body=ARTICLE), time.monotonic() + 100)
    round1 = [o["output"] for o in fake.calls[1]["input"]]
    round2 = [o["output"] for o in fake.calls[2]["input"]]
    assert round1[0].startswith("1. Nvidia Rubin") and round1[1].startswith("1. Nvidia Rubin")
    assert round1[2].startswith("error: search limit reached")
    assert round2[0].startswith(f"URL: {URL}") and round2[1] == "error: tool limit reached; answer with what you have"
    assert len(http.posts) == 2 and f.tools == ["web_search"] * 3 + ["read_article"]
    assert [c.get("tool_choice") for c in fake.calls] == [None, "auto", "auto"] and f.status == "verified"


def test_a_passed_deadline_switches_tools_off_on_the_next_call(http):
    read = ("read_article", {"url": URL})
    llm, fake = _openai(_resp(calls=[read]), _resp(text=_answer([Q1, Q2], URL), calls=[read]))
    r = AgentResearcher(llm, _web("tvly"), [], 30, set(), today=TODAY)
    f = r.investigate(0, _pick(body=ARTICLE), time.monotonic())  # the stage ends now, so this one is past due
    assert [c.get("tool_choice") for c in fake.calls] == [None, "none"]
    assert f.status == "verified" and f.tools == ["read_article"]


# --- all researchers together ---------------------------------------------------------------

def test_eight_researchers_run_in_parallel(http):
    stories, quotes = _eight()
    barrier = threading.Barrier(8, timeout=5)

    def route(name, kw):
        if "previous_response_id" not in kw:
            barrier.wait()  # breaks after 5 s unless all 8 are waiting at once
        i = _index(name)
        return _resp(text=_answer(quotes[i], stories[i].url))

    llm, fake = _routed(route)
    start = time.monotonic()
    AgentResearcher(llm, _web(""), [], 30, set(), today=TODAY).enrich(stories)
    assert time.monotonic() - start < 4 and not barrier.broken
    assert sorted(c["text"]["format"]["name"] for c in fake.calls) == sorted(f"research_{k}" for k in range(1, 9))
    for s, q in zip(stories, quotes):
        assert s.checked == "verified" and [e.quote for e in s.evidence] == q and {e.url for e in s.evidence} == {s.url}


def test_a_crashed_and_a_hung_researcher_keep_the_prefetch(http, monkeypatch):
    monkeypatch.setattr(research, "RESEARCH_STAGE_SECONDS", 1.0)
    stories, quotes = _eight()
    bodies = [s.body for s in stories]
    release = threading.Event()

    def route(name, kw):
        i = _index(name)
        if i == 2:
            return RuntimeError("boom")
        if i == 5:
            release.wait(10)
        return _resp(text=_answer(quotes[i], stories[i].url))

    llm, _ = _routed(route)
    r = AgentResearcher(llm, _web(""), [], 30, set(), today=TODAY)
    start = time.monotonic()
    try:
        r.enrich(stories)
        elapsed = time.monotonic() - start
    finally:
        release.set()
    assert elapsed < 2.5
    for i in (2, 5):
        assert stories[i].checked == "failed" and stories[i].body == bodies[i] and stories[i].evidence == []
    assert r.rows[2]["error"] == "boom" and r.rows[5]["error"] == "timed out"
    for i in (0, 1, 3, 4, 6, 7):
        assert stories[i].checked == "verified" and [e.quote for e in stories[i].evidence] == quotes[i]
    _join_researchers()
    assert stories[5].checked == "failed" and stories[5].evidence == []  # a late answer is not applied


def test_every_researcher_falls_back_from_a_missing_model(http):
    stories, quotes = _eight()
    missing = type("NotFoundError", (Exception,), {"status_code": 404, "code": "model_not_found"})

    def route(name, kw):
        if kw["model"] == "gpt-6-sol":
            return missing("The model gpt-6-sol does not exist")
        i = _index(name)
        return _resp(text=_answer(quotes[i], stories[i].url))

    usage = Usage()
    base, fake = _routed(route, usage=usage)
    llm = base.with_model("gpt-6-sol")
    r = AgentResearcher(llm, _web(""), [], 30, set(), today=TODAY)
    r.enrich(stories)
    assert [s.checked for s in stories] == ["verified"] * 8 and not any(row["error"] for row in r.rows)
    assert Counter(c["model"] for c in fake.calls) == {"gpt-6-sol": 8, "gpt-5": 8}
    assert len(usage.calls) == 8 and {c["model"] for c in usage.calls} == {"gpt-5"}
    assert llm.model == "gpt-6-sol"  # each worker fell back on its own copy


def test_researchers_leave_the_writers_reserve(http):
    stories, _ = _eight()
    http.extract = {s.url: s.body for s in stories}
    for s in stories:
        s.body = ""
    usage = Usage(run_cap_usd=1.0)
    usage.add("editor", "gpt-5", 680_000, 0)  # $0.85, within $0.20 of the cap
    llm, fake = _openai(usage=usage)
    r = AgentResearcher(llm, _web("tvly"), [], 30, set(), today=TODAY)
    r.enrich(stories)
    assert fake.calls == [] and len(usage.calls) == 1
    assert [s.body for s in stories] == [http.extract[s.url] for s in stories]
    assert {s.checked for s in stories} == {"failed"} and all("cap" in row["error"] for row in r.rows)
    usage.check("writer")  # the writer still has room


def test_without_a_tavily_key_research_reads_through_jina(http):
    tracking = "https://links.tldr.tech/abc123"
    http.redirects = {tracking: URL}
    http.jina = {URL: _jina_page(URL, "Nvidia unveils Rubin", _article(Q1, f"Specs are at {BLOG}.")),
                 BLOG: _jina_page(BLOG, "Rubin deep dive", _article(Q3), "2026-09-25T08:00:00Z")}
    llm, fake = _openai(_resp(calls=[("read_article", {"url": BLOG})]),
                        _resp(text=_answer([Q1, Q3], "", first="2026-09-25")))
    story = _pick(url=tracking)
    AgentResearcher(llm, _web(""), [], 30, set(), today=TODAY).enrich([story])
    assert [t["name"] for t in fake.calls[0]["tools"]] == ["read_article"]
    assert http.heads == [tracking] and http.posts == []
    assert http.gets == [JINA_READER + URL, JINA_READER + BLOG]
    assert story.url == URL and story.body.startswith(Q1)
    assert story.checked == "verified" and [e.url for e in story.evidence] == [URL, BLOG]
    assert story.first_reported == "2026-09-25"


def test_a_json_only_llm_gets_the_fixed_step_and_no_calls(http):
    other = "https://openai.example/atlas"
    http.extract = {URL: ARTICLE}
    http.jina = {other: _jina_page(other, "OpenAI Atlas", _article("OpenAI said Atlas can book whole trips."))}
    llm = ScriptedLLM()
    stories = [_pick(), _pick("OpenAI unveils Atlas agent", other)]
    r = AgentResearcher(llm, _web("tvly"), [], 30, set(), today=TODAY)
    r.enrich(stories)
    assert llm.calls == []
    assert stories[0].body == ARTICLE and stories[1].body.startswith("OpenAI said Atlas")
    assert [s.checked for s in stories] == ["", ""] and [row["read"] for row in r.rows] == [True, True]
    assert "quotes" not in r.rows[0]


# --- swaps -----------------------------------------------------------------------------------

class FakeResearcher:
    name = "fake"

    def __init__(self, results=None):
        self.results, self.batches = results or {}, []

    def enrich(self, stories):
        self.batches.append([s.title for s in stories])
        for s in stories:
            s.checked = self.results.get(s.title, "verified")


def _distinct(picks):
    """A pick check that only drops repeated titles."""
    titles = [p.title for p in picks]
    return [p for k, p in enumerate(picks) if p.title not in titles[:k]]


def _checked(title, checked="verified"):
    s = _pick(title, f"https://news.example/{title}", body=f"Body of {title}.")
    s.checked = checked
    return s


def test_swaps_replace_the_worst_two_and_append_the_replacements():
    stories = [_checked("A", "thin"), _checked("B"), _checked("C", "stale"), _checked("D", "wrong_story"),
               _checked("E")]
    bench = [stories[1], _checked("X", ""), _checked("Y", ""), _checked("Z", "")]
    researcher = FakeResearcher()
    out = swap_failing(researcher, stories, bench, _distinct)
    assert researcher.batches == [["X", "Y"]]
    assert [s.title for s in out[:3]] == ["A", "B", "E"] and {s.title for s in out[3:]} == {"X", "Y"}
    assert out[0] is stories[0] and out[0].url and out[0].body  # the third failing story stays as it was
    assert not any(s is b for s in out for b in bench[1:])  # swapped in as copies; the bench is left alone


def test_the_worst_story_gets_the_first_replacement():
    stories = [_checked("A"), _checked("C", "stale"), _checked("D", "wrong_story")]
    bench = [_checked("X", ""), _checked("Y", "")]
    out = swap_failing(FakeResearcher({"X": "wrong_story", "Y": "verified"}), stories, bench, _distinct)
    assert [s.title for s in out] == ["A", "D", "Y"]  # D drew X, which was no better; C drew Y
    assert (out[1].url, out[1].body) == ("", "")


def test_replacements_are_used_only_when_they_rank_higher():
    stories = [_checked("A"), _checked("B", "stale"), _checked("C", "thin")]
    bench = [_checked("X", ""), _checked("Y", "")]
    researcher = FakeResearcher({"X": "stale", "Y": "verified"})
    out = swap_failing(researcher, stories, bench, _distinct)
    assert researcher.batches == [["X", "Y"]]
    assert [s.title for s in out] == ["A", "B", "Y"] and out[1] is stories[1] and out[1].url


def test_a_bench_story_fills_one_slot_at_most():
    stories = [_checked("A"), _checked("C", "stale"), _checked("D", "wrong_story")]
    researcher = FakeResearcher()
    out = swap_failing(researcher, stories, [_checked("X", ""), _checked("Y", "")], lambda picks: picks)
    assert researcher.batches == [["X", "Y"]] and sorted(s.title for s in out) == ["A", "X", "Y"]


def test_a_replacement_that_repeats_any_pick_is_skipped():
    stories = [_checked("A"), _checked("B", "stale"), _checked("C", "thin")]
    again = _checked("B", "")  # the feed article behind pick B: same link, a different object
    researcher = FakeResearcher()
    out = swap_failing(researcher, stories, [again, _checked("X", "")], lambda picks: picks)
    assert researcher.batches == [["X"]]  # B can't replace itself, and C can't take B either
    assert [s.title for s in out] == ["A", "C", "X"]


def test_a_thin_pick_is_kept_unless_the_replacement_is_verified():
    stories = [_checked("A"), _checked("T", "thin"), _checked("S", "stale")]
    out = swap_failing(FakeResearcher({"X": "failed", "Y": "failed"}), stories,
                       [_checked("X", ""), _checked("Y", "")], _distinct)
    assert [s.title for s in out] == ["A", "T", "X"]  # the stale pick goes; the thin one keeps its article
    wrong = [_checked("A"), _checked("W", "wrong_story")]
    out = swap_failing(FakeResearcher({"X": "stale"}), wrong, [_checked("X", "")], _distinct)
    assert [s.title for s in out] == ["A", "W"]  # old news is no better than a bad link


def test_a_kept_wrong_story_loses_its_quotes_and_date_too():
    w = _checked("W", "wrong_story")
    w.evidence, w.first_reported = [research.Evidence(Q1, URL)], "2026-09-20"
    out = swap_failing(FakeResearcher(), [_checked("A"), w], [], _distinct)
    assert (out[1].url, out[1].body, out[1].evidence, out[1].first_reported) == ("", "", [], "")


def test_a_wrong_story_with_no_replacement_loses_its_link():
    stories = [_checked("A"), _checked("D", "wrong_story"), _checked("T", "thin")]
    researcher = FakeResearcher()
    out = swap_failing(researcher, stories, [_checked("X", "")], lambda picks: picks[:-1])  # settle rejects all
    assert researcher.batches == [] and [s.title for s in out] == ["A", "D", "T"]
    assert (out[1].url, out[1].body) == ("", "") and out[2].url and out[2].body
    assert swap_failing(researcher, [_checked("A")], [_checked("X")], lambda picks: picks)[0].title == "A"


def test_swaps_skip_bench_stories_the_pick_check_rejects_and_keep_researched_links(tmp_path):
    kept = _feed_story("OpenAI ships GPT agent")
    wrong, thin = _feed_story("Nvidia unveils AI chip"), _feed_story("EU passes AI rules")
    bench = [_feed_story("OpenAI ships a GPT agent"), _feed_story("Google releases Gemini 3"),
             _feed_story("Google launches Gemini 3.0"), _feed_story("Meta releases Llama 5")]
    candidates = [kept, wrong, thin] + bench
    kept, wrong, thin = (replace(s, headline=s.title) for s in (kept, wrong, thin))
    kept.url, kept.checked = HIT, "verified"  # a source a researcher found, not in today's feeds
    wrong.checked, thin.checked = "wrong_story", "thin"
    seen = SeenStore(tmp_path / "seen.json")
    researcher = FakeResearcher()
    out = swap_failing(researcher, [kept, wrong, thin], bench,
                       lambda picks: settle(picks, candidates, 3, seen, 30))
    assert researcher.batches == [["Google releases Gemini 3", "Meta releases Llama 5"]]
    assert [s.title for s in out] == ["OpenAI ships GPT agent", "Google releases Gemini 3", "Meta releases Llama 5"]
    assert out[0] is kept and kept.url == HIT
    assert out[1].url == bench[1].url and out[2].url == bench[3].url


def test_research_survives_a_crashing_researcher():
    class Boom:
        name = "boom"

        def enrich(self, stories):
            raise RuntimeError("down")

    stories = [_pick(body=ARTICLE)]
    research.research(Boom(), stories)
    assert stories[0].body == ARTICLE


# --- the fixed step and the factory ---------------------------------------------------------

def test_tavily_researcher_counts_its_credits(http, tmp_path):
    urls = [f"https://lab.ai/story-{k}" for k in range(7)]
    http.extract = {u: ARTICLE for u in urls[:6]}
    credits = TavilyCredits(tmp_path / "tavily.json", today=TODAY)
    stories = [_pick(f"Story {k}", u) for k, u in enumerate(urls)]
    TavilyResearcher("tvly", Tavily("tvly", credits)).enrich(stories)
    assert [s.body for s in stories] == [ARTICLE] * 6 + [""]
    assert len(http.posts) == 1 and http.posts[0][1]["urls"] == urls
    assert credits.run_used == 2 and json.loads((tmp_path / "tavily.json").read_text()) == {"2026-09": 2}


def test_build_researcher_picks_agents_tavily_or_nothing():
    cfg = replace(Config.from_env(), agents=True, tavily_api_key="", max_age_hours=30)
    llm, _ = _openai()
    tools = _web("")
    candidates = [_pick()]
    r = build_researcher(cfg, llm, tools, candidates, seen_urls={"https://www.lab.ai/old/?utm_source=x"})
    assert isinstance(r, AgentResearcher) and r.llm is llm and r.web is tools and r.candidates == candidates
    assert r.aired == {"https://lab.ai/old"} and r.max_age_hours == 30
    assert isinstance(build_researcher(replace(cfg, tavily_api_key="tvly"), llm, tools), AgentResearcher)
    assert build_researcher(replace(cfg, agents=False), llm, tools).llm is None
    assert build_researcher(cfg, ScriptedLLM(), tools).llm is None
    assert build_researcher(cfg, None, tools).llm is None
    credits = TavilyCredits(None)
    t = build_researcher(replace(cfg, tavily_api_key="tvly"), llm, None, credits=credits)
    assert isinstance(t, TavilyResearcher) and t.tavily.api_key == "tvly" and t.tavily.credits is credits
    assert isinstance(build_researcher(cfg, llm), NoResearch)
