import threading
import time
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace
from xml.sax.saxutils import escape, quoteattr

import pytest
import requests

from shorts import coverage, sources
from shorts.coverage import Coverage, event_query

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
HN_URL = "https://hn.algolia.com/api/v1/search"
HEADLINE = "Nvidia unveils Rubin GPU at GTC Paris"
UNKNOWN = {"hn_points": None, "hn_threads": None, "news_outlets": None, "outlets": []}


class _Reply:
    def __init__(self, status=200, content=b"", data=None):
        self.status_code, self.content, self._data = status, content, data

    def json(self):
        if self._data is None:
            raise ValueError("not json")
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeHTTP:
    """Routes GETs to the HN or Google News handler; a handler is a reply, an exception or a callable."""

    def __init__(self):
        self.hn = _Reply(data={"hits": []})
        self.news = _Reply(content=_rss())
        self.calls = []
        self._lock = threading.Lock()

    def get(self, url, params=None, headers=None, timeout=None):
        with self._lock:
            self.calls.append((url, dict(params or {})))
        handler = {HN_URL: self.hn, coverage.GOOGLE_NEWS_RSS: self.news}.get(url)
        reply = handler(params) if callable(handler) else handler
        if isinstance(reply, Exception):
            raise reply
        return reply

    def count(self, url):
        return sum(1 for u, _ in self.calls if u == url)


@pytest.fixture
def http(monkeypatch):
    fake = FakeHTTP()
    namespace = SimpleNamespace(get=fake.get, RequestException=requests.RequestException)
    monkeypatch.setattr(coverage, "requests", namespace)
    monkeypatch.setattr(sources, "requests", namespace)
    return fake


def _item(title, hours_ago=2, source=None):
    source_tag = f"<source url={quoteattr('https://' + source.lower().replace(' ', '') + '.com')}>" \
                 f"{escape(source)}</source>" if source else ""
    date = format_datetime(NOW - timedelta(hours=hours_ago), usegmt=True)
    return f"<item><title>{escape(title)}</title><link>https://news.google.com/x</link>{source_tag}" \
           f"<pubDate>{date}</pubDate></item>"


def _rss(*items):
    return (f'<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>Google News</title>'
            f'{"".join(items)}</channel></rss>').encode()


def _hit(title, points, url=None):
    return {"title": title, "points": points, "url": url, "objectID": title[:8]}


# --- event_query ----------------------------------------------------------------------------

def test_event_query_keeps_headline_order_and_drops_filler():
    assert event_query("OpenAI and Microsoft sign a new $10B deal for Stargate") == \
        "OpenAI Microsoft $10B deal Stargate"
    assert event_query("OpenAI and Microsoft sign a new $10B deal for Stargate", max_words=3) == \
        "OpenAI Microsoft $10B"
    assert event_query(HEADLINE) == "Nvidia Rubin GPU GTC Paris"


def test_event_query_of_filler_only_is_empty_and_makes_no_http(http):
    assert event_query("The new AI launch is here") == ""
    assert event_query("") == ""
    row = Coverage(now=NOW).lookup("The new AI launch is here")
    assert http.calls == []
    assert {k: row[k] for k in UNKNOWN} == UNKNOWN


# --- Google News ----------------------------------------------------------------------------

def test_google_news_counts_distinct_matching_outlets_in_the_window(http):
    http.news = _Reply(content=_rss(
        _item("Nvidia unveils Rubin GPU at GTC Paris - Reuters", 2, source="Reuters"),
        _item("Nvidia's Rubin GPU debuts at GTC Paris - The Verge", 3),
        _item("Nvidia unveils Rubin GPU at GTC Paris - Reuters", 1, source="Reuters"),
        _item("Apple sues Samsung over foldable patents - Bloomberg", 2, source="Bloomberg"),
        _item("Nvidia Rubin GPU shown at GTC Paris - CNBC", 72, source="CNBC"),
        _item("Nvidia Rubin GPU arrives at GTC Paris - TechCrunch", 47, source="TechCrunch"),
        _item("Nvidia Rubin GPU at GTC Paris - hands-on", 5, source="Ars Technica"),
        _item("Rubin GPU and the Nvidia GTC Paris keynote - Wired", 50),
    ))
    row = Coverage(now=NOW).lookup(HEADLINE)
    assert row["outlets"] == ["Reuters", "The Verge", "TechCrunch", "Ars Technica"]
    assert row["news_outlets"] == 4
    params = [p for u, p in http.calls if u == coverage.GOOGLE_NEWS_RSS][0]
    assert params["q"] == "Nvidia Rubin GPU GTC Paris when:2d"


def test_google_news_counts_every_outlet_but_lists_at_most_six(http):
    names = [f"Outlet {c}" for c in "ABCDEFGH"]
    http.news = _Reply(content=_rss(*[_item(f"{HEADLINE} - {n}", i + 1, source=n) for i, n in enumerate(names)]))
    row = Coverage(now=NOW).lookup(HEADLINE)
    assert row["news_outlets"] == 8
    assert row["outlets"] == names[:6]


def test_google_news_with_no_matching_item_is_zero_not_unknown(http):
    http.news = _Reply(content=_rss(_item("Apple sues Samsung over foldable patents - Bloomberg")))
    row = Coverage(now=NOW).lookup(HEADLINE)
    assert row["news_outlets"] == 0
    assert row["outlets"] == []


@pytest.mark.parametrize("reply", [
    _Reply(status=429),
    _Reply(status=503),
    _Reply(content=b"<!doctype html><html><body>Before you continue to Google</body></html>"),
])
def test_rate_limit_or_html_turns_google_news_off_with_no_further_calls(http, reply):
    http.news = reply
    cov = Coverage(now=NOW)
    first = cov.lookup(HEADLINE)
    second = cov.lookup("Anthropic ships Claude Opus 5 with 1M context")
    assert first["news_outlets"] is None and second["news_outlets"] is None
    assert http.count(coverage.GOOGLE_NEWS_RSS) == 1
    assert "news" in cov.off and "hn" not in cov.off
    assert http.count(HN_URL) == 2
    assert second["hn_points"] == 0


def test_google_news_network_error_is_unknown(http):
    http.news = requests.ConnectionError("down")
    row = Coverage(now=NOW).lookup(HEADLINE)
    assert row["news_outlets"] is None
    assert row["outlets"] == []
    assert row["hn_points"] == 0


# --- Hacker News ----------------------------------------------------------------------------

def test_hacker_news_takes_the_highest_points_among_matching_hits(http):
    http.hn = _Reply(data={"hits": [
        _hit("Nvidia unveils Rubin GPU at GTC Paris", 120, "https://nvidianews.nvidia.com/rubin"),
        _hit("Nvidia Rubin GPU GTC Paris keynote", 300, "https://youtube.com/watch?v=1"),
        _hit("Show HN: My weekend project", 900, "https://github.com/me/project"),
        _hit("Ask HN: Who is hiring?", 700),
    ]})
    row = Coverage(now=NOW).lookup(HEADLINE)
    assert row["hn_points"] == 300
    assert row["hn_threads"] == 2


def test_hacker_news_matches_a_hit_by_the_same_url(http):
    http.hn = _Reply(data={"hits": [
        _hit("Nvidia unveils Rubin GPU at GTC Paris", 120),
        _hit("Discussion thread", 450, "https://www.nvidianews.nvidia.com/rubin/?utm_source=hn"),
        _hit("Unrelated post", 800, "https://nvidianews.nvidia.com/other"),
    ]})
    row = Coverage(now=NOW).lookup(HEADLINE, url="https://nvidianews.nvidia.com/rubin")
    assert row["hn_points"] == 450
    assert row["hn_threads"] == 2


def test_hacker_news_working_search_with_no_match_gives_zero(http):
    http.hn = _Reply(data={"hits": [_hit("Show HN: My weekend project", 900, "https://github.com/me/p"),
                                     _hit("Ask HN: Who is hiring?", 700)]})
    row = Coverage(now=NOW).lookup(HEADLINE)
    assert row["hn_points"] == 0
    assert row["hn_threads"] == 0


def test_hacker_news_request_uses_the_query_and_the_window(http):
    Coverage(max_age_hours=30, now=NOW).lookup(HEADLINE)
    Coverage(max_age_hours=72, now=NOW).lookup(HEADLINE)
    params = [p for u, p in http.calls if u == HN_URL]
    assert [p["query"] for p in params] == ["Nvidia Rubin GPU GTC Paris"] * 2
    assert params[0]["tags"] == "story"
    assert f"created_at_i>{int((NOW - timedelta(hours=48)).timestamp())}," in params[0]["numericFilters"]
    assert f"created_at_i>{int((NOW - timedelta(hours=72)).timestamp())}," in params[1]["numericFilters"]


@pytest.mark.parametrize("reply", [_Reply(status=500), _Reply(content=b"<html>", data=None),
                                   requests.Timeout("slow")])
def test_hacker_news_errors_are_unknown_not_zero(http, reply):
    http.hn = reply
    http.news = _Reply(content=_rss(_item(f"{HEADLINE} - Reuters", source="Reuters")))
    row = Coverage(now=NOW).lookup(HEADLINE)
    assert row["hn_points"] is None
    assert row["hn_threads"] is None
    assert row["news_outlets"] == 1


def test_hn_search_sends_params_and_raises_on_http_errors(http):
    since = NOW - timedelta(hours=10)
    http.hn = _Reply(data={"hits": [_hit("A", 5)]})
    assert sources.hn_search("GPT-5", since, min_points=50, hits=7) == [_hit("A", 5)]
    url, params = http.calls[-1]
    assert url == HN_URL
    assert params == {"query": "GPT-5", "tags": "story", "hitsPerPage": 7,
                      "numericFilters": f"created_at_i>{int(since.timestamp())},points>50"}
    http.hn = _Reply(status=502)
    with pytest.raises(requests.HTTPError):
        sources.hn_search("GPT-5", since)


# --- cache, cap and parallel lookups --------------------------------------------------------

def test_lookups_are_cached(http):
    http.hn = _Reply(data={"hits": [_hit(HEADLINE, 120)]})
    cov = Coverage(now=NOW)
    first = cov.lookup(HEADLINE)
    calls = len(http.calls)
    assert calls == 2
    assert cov.lookup(HEADLINE) == first
    assert len(http.calls) == calls
    assert cov.rows() == [first]


def test_request_cap_holds_per_backend(http):
    cov = Coverage(max_requests=2, now=NOW)
    rows = [cov.lookup(h) for h in ("Nvidia unveils Rubin GPU", "Anthropic ships Claude Opus",
                                    "Mistral releases Magistral Medium")]
    assert http.count(HN_URL) == 2
    assert http.count(coverage.GOOGLE_NEWS_RSS) == 2
    assert cov.requests == {"hn": 2, "news": 2}
    assert rows[1]["hn_points"] == 0 and rows[1]["news_outlets"] == 0
    assert {k: rows[2][k] for k in UNKNOWN} == UNKNOWN


def test_lookup_many_returns_unknown_rows_for_lookups_that_time_out(http):
    gate = threading.Event()
    slow = "Stalled Mistral Magistral benchmark leak"

    def hn(params):
        if params["query"].startswith("Stalled"):
            gate.wait(5)
        return _Reply(data={"hits": [_hit(HEADLINE, 120)]})

    http.hn = hn
    try:
        start = time.monotonic()
        rows = Coverage(now=NOW).lookup_many([HEADLINE, slow], timeout=0.3)
        elapsed = time.monotonic() - start
    finally:
        gate.set()
    assert elapsed < 2
    assert set(rows) == {HEADLINE, slow}
    assert rows[slow] == {"headline": slow, **UNKNOWN}
    assert rows[HEADLINE]["hn_points"] == 120
    assert rows[HEADLINE]["news_outlets"] == 0


def test_coverage_line_formats_known_and_unknown_values():
    assert Coverage.line({"hn_points": 120, "news_outlets": 4}) == "HN 120 pts, 4 outlets on Google News"
    assert Coverage.line({"hn_points": 0, "news_outlets": 0}) == "HN 0 pts, 0 outlets on Google News"
    assert Coverage.line({"hn_points": None, "news_outlets": None}) == "HN unknown, Google News unknown"
    assert Coverage.line({}) == "HN unknown, Google News unknown"
    assert Coverage.line({"hn_points": 7}) == "HN 7 pts, Google News unknown"
