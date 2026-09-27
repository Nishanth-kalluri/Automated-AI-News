import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from shorts import coverage, web
from shorts.config import Config
from shorts.coverage import Coverage
from shorts.web import (CreditsExhausted, Doc, Jina, Tavily, TavilyCredits, Web, WebUnavailable, build_web, iso_date,
                        junk_reason, normalize, parse_jina, quote_in)

SEP26 = date(2026, 9, 26)  # 5 days left in September, counting today
ARTICLE = "OpenAI released GPT-6 on Thursday, calling it its most capable model so far. " * 12


class FakeHTTP:
    """Stands in for ``requests`` inside shorts.web: replies in order and records every call."""

    RequestException = requests.RequestException

    def __init__(self, *replies, on_call=None):
        self.replies, self.calls, self.on_call = list(replies), [], on_call

    def _call(self, method, url, **kw):
        call = SimpleNamespace(method=method, url=url, json=kw.get("json"), headers=kw.get("headers") or {},
                               timeout=kw.get("timeout"))
        self.calls.append(call)
        if self.on_call:
            self.on_call(call)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def post(self, url, **kw):
        return self._call("post", url, **kw)

    def get(self, url, **kw):
        return self._call("get", url, **kw)


def _http(status=200, data=None, text=""):
    def js():
        if isinstance(data, Exception):
            raise data
        return data
    return SimpleNamespace(status_code=status, json=js, text=text)


def _credits(tmp_path, used=None, month_cap=700, run_cap=25, today=SEP26):
    path = tmp_path / "tavily.json"
    if used is not None:
        path.write_text(json.dumps(used))
    return TavilyCredits(path, month_cap, run_cap, today=today)


def _saved(tmp_path):
    return json.loads((tmp_path / "tavily.json").read_text())


def _tavily(monkeypatch, tmp_path, *replies, credits=None):
    fake = FakeHTTP(*replies)
    monkeypatch.setattr(web, "requests", fake)
    return Tavily("tvly-key", credits or _credits(tmp_path)), fake


def _hit(url="https://openai.com/index/gpt-6", **kw):
    return {"title": "OpenAI ships GPT-6", "url": url, "content": "OpenAI released GPT-6.", **kw}


def _jina_body(text=ARTICLE, title="GPT-6: what's new", source="https://openai.com/index/gpt-6",
               published="2026-09-25T08:30:00.000Z"):
    return f"Title: {title}\n\nURL Source: {source}\n\nPublished Time: {published}\n\nMarkdown Content:\n{text}\n"


# --- credits --------------------------------------------------------------------------------

def test_credits_charge_within_the_allowance_then_refuse(tmp_path):
    credits = _credits(tmp_path)
    assert credits.allowance == 25  # min(25, 700 // 5)
    credits.charge(20)
    with pytest.raises(CreditsExhausted):
        credits.charge(6)
    assert credits.run_used == 20 and credits.this_month() == 20 and credits.remaining == 5
    credits.charge(5)
    assert credits.remaining == 0
    with pytest.raises(CreditsExhausted):
        credits.charge(1)
    assert credits.run_used == 25 and credits.this_month() == 25 and _saved(tmp_path) == {"2026-09": 25}


@pytest.mark.parametrize("today, used, run_cap, allowance", [
    (SEP26, 650, 25, 10),  # 50 left over 5 days
    (SEP26, 650, 8, 8),  # the run cap is lower
    (date(2026, 9, 1), 0, 25, 23),  # 700 // 30
    (date(2026, 9, 30), 690, 25, 10),  # the last day may spend all that is left
    (date(2026, 2, 1), 0, 100, 25),  # 700 // 28
    (SEP26, 800, 25, 0),  # already over the cap
])
def test_credits_pace_the_month(tmp_path, today, used, run_cap, allowance):
    credits = _credits(tmp_path, {today.strftime("%Y-%m"): used}, run_cap=run_cap, today=today)
    assert credits.allowance == allowance
    assert credits.remaining == allowance


def test_credits_new_run_resets_and_repaces(tmp_path):
    credits = _credits(tmp_path, {"2026-09": 650})
    credits.charge(10)
    assert credits.remaining == 0
    credits.new_run()
    assert credits.run_used == 0 and credits.allowance == 8  # 40 left over 5 days
    credits.charge(8)
    assert _saved(tmp_path) == {"2026-09": 668}


def test_credits_roll_over_to_a_new_month(tmp_path):
    credits = _credits(tmp_path, {"2026-08": 700}, today=date(2026, 9, 1))
    assert credits.month == "2026-09" and credits.this_month() == 0 and credits.allowance == 23
    credits.charge(3)
    assert _saved(tmp_path) == {"2026-08": 700, "2026-09": 3}
    credits.today = date(2026, 10, 31)
    credits.new_run()
    assert credits.month == "2026-10" and credits.this_month() == 0 and credits.allowance == 25
    credits.charge(1)
    assert _saved(tmp_path) == {"2026-08": 700, "2026-09": 3, "2026-10": 1}


def test_credits_write_through_to_the_state_file(tmp_path):
    credits = _credits(tmp_path)
    assert not (tmp_path / "tavily.json").exists()
    credits.charge(1)
    assert _saved(tmp_path) == {"2026-09": 1}
    credits.charge(2)
    assert _saved(tmp_path) == {"2026-09": 3}
    assert _credits(tmp_path).this_month() == 3
    assert [p.name for p in tmp_path.iterdir()] == ["tavily.json"]


@pytest.mark.parametrize("content", [b"{not json", b"[1, 2]", b'"text"', b'{"2026-09": "lots"}',
                                     b'{"2026-09": null}', b"\xff\xfe\x00junk"])
def test_credits_unreadable_file_starts_at_zero(tmp_path, content):
    (tmp_path / "tavily.json").write_bytes(content)
    credits = TavilyCredits(tmp_path / "tavily.json", today=SEP26)
    assert credits.months == {} and credits.this_month() == 0 and credits.allowance == 25
    credits.charge(2)
    assert _saved(tmp_path) == {"2026-09": 2}


def test_credits_file_that_cannot_be_opened_starts_at_zero(tmp_path, monkeypatch):
    (tmp_path / "tavily.json").write_text('{"2026-09": 5}')

    def denied(self, *a, **kw):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "read_text", denied)
    credits = TavilyCredits(tmp_path / "tavily.json", today=SEP26)
    assert credits.this_month() == 0 and credits.allowance == 25


def test_credits_exhaust_uses_up_the_month(tmp_path):
    credits = _credits(tmp_path, {"2026-08": 5, "2026-09": 100})
    credits.charge(2)
    credits.exhaust()
    assert credits.this_month() == 700 and credits.remaining == 0
    assert _saved(tmp_path) == {"2026-08": 5, "2026-09": 700}
    with pytest.raises(CreditsExhausted):
        credits.charge(1)
    later = _credits(tmp_path)  # a later run this month
    assert later.allowance == 0 and later.remaining == 0


def test_credits_without_a_file_count_without_caps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    credits = TavilyCredits(None, month_cap=1, run_cap=1, today=SEP26)
    for _ in range(3):
        credits.charge(5)
    assert credits.run_used == 15 and credits.this_month() == 15 and credits.remaining == math.inf
    credits.new_run()
    assert credits.run_used == 0 and credits.allowance == math.inf
    assert list(tmp_path.iterdir()) == []


def _charge_in_threads(credits, threads=8, each=50):
    start, lock, refused = threading.Barrier(threads, timeout=5), threading.Lock(), []

    def work():
        start.wait()
        for _ in range(each):
            try:
                credits.charge(1)
            except CreditsExhausted:
                with lock:
                    refused.append(1)

    pool = [threading.Thread(target=work) for _ in range(threads)]
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # switch threads often, so a missing lock shows
    try:
        for t in pool:
            t.start()
        for t in pool:
            t.join(10)
    finally:
        sys.setswitchinterval(interval)
    return len(refused)


def test_credits_are_exact_under_threads(tmp_path):
    credits = _credits(tmp_path, month_cap=10_000, run_cap=1000, today=date(2026, 9, 30))
    assert _charge_in_threads(credits) == 0
    assert credits.run_used == 400 and credits.this_month() == 400
    assert _saved(tmp_path) == {"2026-09": 400}
    assert [p.name for p in tmp_path.iterdir()] == ["tavily.json"]


def test_credits_allowance_holds_under_threads(tmp_path):
    credits = _credits(tmp_path, month_cap=10_000, run_cap=100, today=date(2026, 9, 30))
    assert _charge_in_threads(credits) == 300
    assert credits.run_used == 100 and _saved(tmp_path) == {"2026-09": 100}


# --- Tavily ---------------------------------------------------------------------------------

def test_iso_date_reads_iso_and_rfc_2822():
    assert iso_date("2026-09-25T08:30:00Z") == "2026-09-25"
    assert iso_date("2026-09-25") == "2026-09-25"
    assert iso_date("Thu, 24 Sep 2026 10:00:00 GMT") == "2026-09-24"
    assert iso_date("yesterday") == iso_date("") == iso_date(None) == ""


def test_tavily_search_payload_header_and_docs(monkeypatch, tmp_path):
    raw = "word " * 2000
    data = {"results": [
        _hit(content="c" * 400, raw_content=raw, published_date="2026-09-25T08:30:00Z"),
        {"title": None, "url": "https://news.example/b", "content": "Short snippet",
         "published_date": "Thu, 24 Sep 2026 10:00:00 GMT"},
        {"url": "https://news.example/c"},
        {"title": "Not a web link", "url": "ftp://files.example/x", "content": "x"},
        "junk",
    ]}
    tavily, fake = _tavily(monkeypatch, tmp_path, _http(200, data))
    at_call = []
    fake.on_call = lambda call: at_call.append((tavily.credits.run_used, _saved(tmp_path)))
    docs = tavily.search("q" * 500)
    call = fake.calls[0]
    assert call.method == "post" and call.url == "https://api.tavily.com/search"
    assert call.json == {"query": "q" * 400, "search_depth": "basic", "topic": "general", "time_range": "week",
                         "max_results": 5, "include_raw_content": True}
    assert call.headers["Authorization"] == "Bearer tvly-key" and call.timeout == 20
    assert at_call == [(1, {"2026-09": 1})]  # charged and saved before the HTTP call
    assert docs == [
        Doc(url="https://openai.com/index/gpt-6", title="OpenAI ships GPT-6", text=raw[:6000], snippet="c" * 300,
            published="2026-09-25", via="search"),
        Doc(url="https://news.example/b", title="", text="Short snippet", snippet="Short snippet",
            published="2026-09-24", via="search"),
        Doc(url="https://news.example/c", via="search"),
    ]
    assert tavily.credits.run_used == 1


def test_tavily_search_passes_max_results(monkeypatch, tmp_path):
    tavily, fake = _tavily(monkeypatch, tmp_path, _http(200, {"results": []}))
    assert tavily.search("OpenAI GPT-6", 3) == []
    assert fake.calls[0].json["max_results"] == 3


def test_tavily_crash_mid_call_still_counts(monkeypatch, tmp_path):
    tavily, fake = _tavily(monkeypatch, tmp_path, requests.ConnectionError("reset"))
    with pytest.raises(WebUnavailable):
        tavily.search("x")
    assert tavily.credits.run_used == 1 and _saved(tmp_path) == {"2026-09": 1}


def test_tavily_400_retries_once_with_minimal_fields_and_stays_minimal(monkeypatch, tmp_path):
    tavily, fake = _tavily(monkeypatch, tmp_path, _http(400, {"detail": "bad field"}),
                           _http(200, {"results": [_hit()]}), _http(200, {"results": []}),
                           _http(200, {"results": [{"url": "https://a.example/1", "raw_content": "Body"}]}))
    docs = tavily.search("OpenAI GPT-6", 3)
    assert [d.url for d in docs] == ["https://openai.com/index/gpt-6"]
    assert [c.json for c in fake.calls] == [
        {"query": "OpenAI GPT-6", "search_depth": "basic", "topic": "general", "time_range": "week",
         "max_results": 3, "include_raw_content": True},
        {"query": "OpenAI GPT-6", "max_results": 3},
    ]
    assert fake.calls[1].headers["Authorization"] == "Bearer tvly-key"
    assert tavily.minimal and not tavily.disabled and tavily.credits.run_used == 1
    tavily.search("Nvidia earnings", 5)
    assert fake.calls[2].json == {"query": "Nvidia earnings", "max_results": 5}
    assert tavily.extract(["https://a.example/1"]) == {"https://a.example/1": Doc(url="https://a.example/1",
                                                                                   text="Body", via="tavily")}
    assert fake.calls[3].json == {"urls": ["https://a.example/1"]}
    assert len(fake.calls) == 4


def test_tavily_400_on_the_minimal_payload_is_not_retried(monkeypatch, tmp_path):
    tavily, fake = _tavily(monkeypatch, tmp_path, _http(400), _http(400), _http(400))
    with pytest.raises(WebUnavailable):
        tavily.search("x")
    assert len(fake.calls) == 2
    with pytest.raises(WebUnavailable):
        tavily.search("y")
    assert len(fake.calls) == 3 and fake.calls[2].json == {"query": "y", "max_results": 5}


@pytest.mark.parametrize("status", [432, 433])
def test_tavily_plan_limit_disables_and_exhausts_the_month(monkeypatch, tmp_path, status):
    tavily, fake = _tavily(monkeypatch, tmp_path, _http(status, {"detail": "plan limit"}))
    with pytest.raises(WebUnavailable):
        tavily.search("x")
    assert tavily.disabled and not tavily.usable
    assert _saved(tmp_path) == {"2026-09": 700} and tavily.credits.remaining == 0
    with pytest.raises(WebUnavailable):
        tavily.search("y")
    with pytest.raises(WebUnavailable):
        tavily.extract(["https://a.example/1"])
    assert len(fake.calls) == 1
    assert _credits(tmp_path).allowance == 0  # later runs this month skip Tavily


@pytest.mark.parametrize("status", [401, 403, 429])
def test_tavily_bad_key_or_rate_limit_disables_for_the_run(monkeypatch, tmp_path, status):
    tavily, fake = _tavily(monkeypatch, tmp_path, _http(status))
    with pytest.raises(WebUnavailable):
        tavily.search("x")
    assert tavily.disabled and not tavily.usable
    with pytest.raises(WebUnavailable):
        tavily.search("y")
    with pytest.raises(WebUnavailable):
        tavily.extract(["https://a.example/1"])
    assert len(fake.calls) == 1
    assert tavily.credits.run_used == 1 and _saved(tmp_path) == {"2026-09": 1}  # the month is not marked used up


def test_tavily_two_server_errors_in_a_row_disable_it(monkeypatch, tmp_path):
    tavily, fake = _tavily(monkeypatch, tmp_path, _http(500), _http(503))
    with pytest.raises(WebUnavailable):
        tavily.search("x")
    assert not tavily.disabled and tavily.usable
    with pytest.raises(WebUnavailable):
        tavily.search("y")
    assert tavily.disabled and not tavily.usable
    with pytest.raises(WebUnavailable):
        tavily.search("z")
    assert len(fake.calls) == 2


def test_tavily_timeouts_count_as_failures(monkeypatch, tmp_path):
    tavily, fake = _tavily(monkeypatch, tmp_path, requests.Timeout("slow"), requests.ConnectionError("down"))
    for query in ("x", "y", "z"):
        with pytest.raises(WebUnavailable):
            tavily.search(query)
    assert tavily.disabled and len(fake.calls) == 2 and tavily.credits.run_used == 2


def test_tavily_a_success_resets_the_failure_count(monkeypatch, tmp_path):
    tavily, fake = _tavily(monkeypatch, tmp_path, _http(502), _http(200, {"results": []}), _http(500),
                           _http(200, {"results": [_hit()]}))
    with pytest.raises(WebUnavailable):
        tavily.search("a")
    assert tavily.search("b") == []
    with pytest.raises(WebUnavailable):
        tavily.search("c")
    assert not tavily.disabled
    assert len(tavily.search("d")) == 1


@pytest.mark.parametrize("data", [requests.exceptions.JSONDecodeError("Expecting value", "<html>", 0), ["a list"],
                                  {"results": None}, {"results": "junk"}, {"answer": "no results key"}])
def test_tavily_junk_json_gives_no_docs(monkeypatch, tmp_path, data):
    tavily, fake = _tavily(monkeypatch, tmp_path, _http(200, data, text="<html>oops</html>"),
                           _http(200, data, text="<html>oops</html>"))
    assert tavily.search("x") == []
    assert tavily.extract(["https://a.example/1"]) == {}
    assert not tavily.disabled and len(fake.calls) == 2


def test_tavily_extract_charges_per_five_links_and_maps_results(monkeypatch, tmp_path):
    urls = [f"https://news.example/{i}" for i in range(8)]
    data = {"results": [{"url": urls[0], "raw_content": "Body zero"}, {"url": urls[3], "raw_content": "Body three"},
                        {"url": urls[4], "raw_content": ""}, {"url": urls[5]}, {"raw_content": "no url"}, "junk"],
            "failed_results": [{"url": urls[1], "error": "timeout"}, {"url": urls[2]}, "junk"]}
    tavily, fake = _tavily(monkeypatch, tmp_path, _http(200, data))
    at_call = []
    fake.on_call = lambda call: at_call.append(tavily.credits.run_used)
    out = tavily.extract(urls + ["mailto:press@example.com"])
    call = fake.calls[0]
    assert call.url == "https://api.tavily.com/extract"
    assert call.json == {"urls": urls, "extract_depth": "basic", "format": "text"}
    assert call.headers["Authorization"] == "Bearer tvly-key"
    assert at_call == [2] and _saved(tmp_path) == {"2026-09": 2}
    assert out == {urls[0]: Doc(url=urls[0], text="Body zero", via="tavily"),
                   urls[3]: Doc(url=urls[3], text="Body three", via="tavily")}


@pytest.mark.parametrize("count, charged, sent", [(1, 1, 1), (5, 1, 5), (6, 2, 6), (25, 4, 20)])
def test_tavily_extract_cost(monkeypatch, tmp_path, count, charged, sent):
    tavily, fake = _tavily(monkeypatch, tmp_path, _http(200, {"results": []}))
    tavily.extract([f"https://news.example/{i}" for i in range(count)])
    assert tavily.credits.run_used == charged and len(fake.calls[0].json["urls"]) == sent


def test_a_batched_extract_waits_as_long_as_before_and_one_link_does_not(monkeypatch, tmp_path):
    tavily, fake = _tavily(monkeypatch, tmp_path, _http(200, {"results": []}), _http(200, {"results": []}),
                           _http(200, {"results": []}))
    tavily.extract([f"https://news.example/{i}" for i in range(8)])
    tavily.extract(["https://news.example/one"])
    tavily.search("rubin")
    assert [c.timeout for c in fake.calls] == [90, 20, 20]


def test_tavily_extract_of_no_links_makes_no_call(monkeypatch, tmp_path):
    tavily, fake = _tavily(monkeypatch, tmp_path)
    assert tavily.extract([]) == {} and tavily.extract(["mailto:x@example.com", ""]) == {}
    assert fake.calls == [] and tavily.credits.run_used == 0


def test_tavily_out_of_credits_raises_without_http(monkeypatch, tmp_path):
    tavily, fake = _tavily(monkeypatch, tmp_path, credits=_credits(tmp_path, {"2026-09": 700}))
    assert not tavily.usable
    with pytest.raises(WebUnavailable, match="credit limit"):
        tavily.search("x")
    with pytest.raises(WebUnavailable, match="credit limit"):
        tavily.extract(["https://a.example/1"])
    assert fake.calls == [] and _saved(tmp_path) == {"2026-09": 700}


def test_tavily_stops_when_the_run_allowance_is_spent(monkeypatch, tmp_path):
    credits = _credits(tmp_path, {"2026-09": 695})  # 5 left over 5 days: 1 this run
    tavily, fake = _tavily(monkeypatch, tmp_path, _http(200, {"results": [_hit()]}), credits=credits)
    assert credits.allowance == 1 and tavily.usable
    with pytest.raises(WebUnavailable):
        tavily.extract([f"https://news.example/{i}" for i in range(8)])  # needs 2
    assert fake.calls == [] and credits.run_used == 0
    assert len(tavily.search("x")) == 1
    assert not tavily.usable
    with pytest.raises(WebUnavailable):
        tavily.search("y")
    assert len(fake.calls) == 1 and _saved(tmp_path) == {"2026-09": 696}


def test_tavily_without_a_key_is_unusable(monkeypatch, tmp_path):
    fake = FakeHTTP()
    monkeypatch.setattr(web, "requests", fake)
    tavily = Tavily("", _credits(tmp_path))
    assert not tavily.usable
    with pytest.raises(WebUnavailable):
        tavily.search("x")
    assert fake.calls == [] and tavily.credits.run_used == 0


# --- Jina -----------------------------------------------------------------------------------

def test_parse_jina_reads_the_header_lines():
    doc = parse_jina("https://t.co/abc", _jina_body())
    assert doc == Doc(url="https://openai.com/index/gpt-6", title="GPT-6: what's new", text=ARTICLE.strip(),
                      published="2026-09-25", via="jina")
    assert parse_jina("https://t.co/abc", _jina_body(published="Thu, 24 Sep 2026 10:00:00 GMT")).published == \
        "2026-09-24"


def test_parse_jina_without_headers_uses_the_whole_body():
    doc = parse_jina("https://news.example/a", "  Title: not a header block\n" + "x" * 7000)
    assert doc.url == "https://news.example/a" and doc.title == "" and doc.published == ""
    assert doc.text.startswith("Title: not a header block") and len(doc.text) == 6000


def test_junk_reason_flags_short_and_paywalled_text():
    assert junk_reason(ARTICLE) is None
    assert junk_reason("x" * 499) == junk_reason(" " * 800) == junk_reason("") == junk_reason(None) == "too short"
    assert junk_reason("SUBSCRIBE TO CONTINUE reading. " + ARTICLE) == "says 'subscribe to continue'"
    assert junk_reason(ARTICLE + "Are you a robot?") == "says 'are you a robot'"


def test_jina_read_fetches_through_the_reader(monkeypatch):
    fake = FakeHTTP(_http(200, text=_jina_body()))
    monkeypatch.setattr(web, "requests", fake)
    jina = Jina()
    doc = jina.read("https://openai.com/index/gpt-6")
    call = fake.calls[0]
    assert call.method == "get" and call.url == "https://r.jina.ai/https://openai.com/index/gpt-6"
    assert call.headers["Accept"] == "text/plain" and "Authorization" not in call.headers and call.timeout == 20
    assert doc == Doc(url="https://openai.com/index/gpt-6", title="GPT-6: what's new", text=ARTICLE.strip(),
                      published="2026-09-25", via="jina")
    assert jina.calls == 1


@pytest.mark.parametrize("text", ["Too short to be an article.", "Subscribe to continue reading. " + ARTICLE,
                                  ARTICLE + " Please enable JavaScript to view this page.",
                                  "Checking your browser: complete the CAPTCHA. " * 20])
def test_jina_rejects_short_and_paywalled_pages(monkeypatch, text):
    fake = FakeHTTP(_http(200, text=_jina_body(text)))
    monkeypatch.setattr(web, "requests", fake)
    jina = Jina()
    assert jina.read("https://news.example/a") is None
    assert len(fake.calls) == 1 and not jina.disabled


def test_jina_errors_give_none_and_keep_it_on(monkeypatch):
    fake = FakeHTTP(_http(404, text="not found"), requests.Timeout("slow"), _http(200, text=_jina_body()))
    monkeypatch.setattr(web, "requests", fake)
    jina = Jina()
    assert jina.read("https://news.example/a") is None
    assert jina.read("https://news.example/b") is None
    assert not jina.disabled
    assert jina.read("https://news.example/c").title == "GPT-6: what's new"


def test_jina_429_turns_it_off_for_the_run(monkeypatch):
    fake = FakeHTTP(_http(429, text="rate limited"))
    monkeypatch.setattr(web, "requests", fake)
    jina = Jina()
    assert jina.read("https://news.example/a") is None
    assert jina.disabled
    assert jina.read("https://news.example/b") is None
    assert len(fake.calls) == 1


def test_jina_call_budget_holds(monkeypatch):
    fake = FakeHTTP(_http(200, text=_jina_body()), _http(500))
    monkeypatch.setattr(web, "requests", fake)
    jina = Jina(max_calls=2)
    assert jina.read("https://news.example/a") is not None
    assert jina.read("https://news.example/b") is None
    assert jina.read("https://news.example/c") is None
    assert len(fake.calls) == 2 and jina.calls == 2


def test_jina_skips_non_web_links(monkeypatch):
    fake = FakeHTTP()
    monkeypatch.setattr(web, "requests", fake)
    jina = Jina()
    assert jina.read("") is None and jina.read("mailto:x@example.com") is None
    assert fake.calls == [] and jina.calls == 0


def test_jina_reads_at_most_three_at_once(monkeypatch):
    lock, state = threading.Lock(), {"now": 0, "peak": 0, "calls": 0}

    def get(url, **kw):
        with lock:
            state["now"] += 1
            state["calls"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.1)
        with lock:
            state["now"] -= 1
        return _http(200, text=_jina_body(source=url.removeprefix("https://r.jina.ai/")))

    monkeypatch.setattr(web, "requests", SimpleNamespace(get=get, RequestException=requests.RequestException))
    jina = Jina()
    urls = [f"https://news.example/{i}" for i in range(8)]
    with ThreadPoolExecutor(8) as pool:
        docs = list(pool.map(jina.read, urls))
    assert state["calls"] == 8 and state["peak"] == 3
    assert [d.url for d in docs] == urls


# --- quote checks ---------------------------------------------------------------------------

TEXT = ("OpenAI said on Thursday that GPT‑6 scored 93.4 % on the “hard” benchmark. "
        "The model is **twice as fast** as GPT-5, according to [the company’s blog](https://openai.com/blog). "
        "It will be available to all ChatGPT users next week, and pricing stays the same.")


def test_normalize_evens_out_quotes_dashes_spaces_and_markdown():
    assert normalize("“Hello,” she said — it’s ‘fine’") == \
        "\"hello,\" she said - it's 'fine'"
    assert normalize("GPT‑5 – state‐of‐the‐art −3") == "gpt-5 - state-of-the-art -3"
    assert normalize("93.4 % and 12 % and 7 %") == "93.4% and 12% and 7%"
    assert normalize("a b c\n\n\t d  ") == "a b c d"
    assert normalize("See [the blog post](https://openai.com/x) and ![chart](https://img.example/x.png)") == \
        "see the blog post and chart"
    assert normalize("**Bold** and _italic_ and `code` ## Heading") == "bold and italic and code heading"
    assert normalize("") == normalize(None) == ""


def test_quote_in_matches_across_typography_and_markdown():
    assert quote_in('GPT-6 scored 93.4% on the "hard" benchmark', TEXT)
    assert quote_in("GPT-6 scored 93.4 % on the “hard” benchmark", TEXT)
    assert quote_in("The model is twice as fast as GPT‑5, according to the company's blog", TEXT)
    assert quote_in("it will be AVAILABLE to all ChatGPT users", TEXT)


def test_quote_in_needs_every_part_in_order():
    assert quote_in("OpenAI said on Thursday ... it will be available to all ChatGPT users", TEXT)
    assert quote_in("OpenAI said on Thursday… pricing stays the same", TEXT)
    assert quote_in("OpenAI said on Thursday ... GPT-6 ... scored 93.4%", TEXT)  # short parts are checked too
    assert not quote_in("Wow ... the model is twice as fast", TEXT)
    assert not quote_in("OpenAI said on Thursday ... $30,000 each ... pricing stays the same", TEXT)
    assert not quote_in("it will be available to all ChatGPT users ... OpenAI said on Thursday", TEXT)
    assert not quote_in("OpenAI said on Thursday ... it will be free for everyone forever", TEXT)
    assert not quote_in("OpenAI said ... it will ...", TEXT)  # no part long enough to check


def test_quote_in_rejects_paraphrases_and_short_quotes():
    assert not quote_in("OpenAI says GPT-6 is two times faster than GPT-5", TEXT)
    assert not quote_in("the model is available to every ChatGPT user", TEXT)
    assert not quote_in("twice as fast", TEXT)
    assert quote_in("twice as fast as", TEXT)
    assert not quote_in("5 million users in the first week", "It had 25 million users in the first week.")
    assert quote_in("25 million users in the first week", "It had 25 million users in the first week.")
    assert not quote_in("", TEXT) and not quote_in("the model is twice as fast", "")


# --- build_web ------------------------------------------------------------------------------

def _cfg(**kw):
    return replace(Config.from_env().offline(), **kw)


def test_build_web_is_off_without_the_flag_or_on_sample_news(monkeypatch):
    fake = FakeHTTP()
    monkeypatch.setattr(web, "requests", fake)
    monkeypatch.setattr(coverage, "requests", fake)
    credits = TavilyCredits(None)
    assert build_web(_cfg(web=False, sources=["rss"], tavily_api_key="tvly"), credits) is None
    assert build_web(_cfg(web=True, sources=["sample"], tavily_api_key="tvly"), credits) is None
    assert fake.calls == []


def test_build_web_without_a_key_has_no_tavily(monkeypatch):
    fake = FakeHTTP()
    monkeypatch.setattr(web, "requests", fake)
    monkeypatch.setattr(coverage, "requests", fake)
    credits = TavilyCredits(None)
    tools = build_web(_cfg(web=True, sources=["rss"], tavily_api_key=""), credits)
    assert isinstance(tools, Web) and tools.tavily is None
    assert isinstance(tools.jina, Jina) and isinstance(tools.coverage, Coverage)
    tools = build_web(_cfg(web=True, sources=["rss", "hackernews"], tavily_api_key="tvly", max_age_hours=72),
                      credits)
    assert isinstance(tools.tavily, Tavily) and tools.tavily.api_key == "tvly" and tools.tavily.credits is credits
    assert tools.coverage.now - tools.coverage.since == timedelta(hours=72)
    assert fake.calls == []
