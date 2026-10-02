"""What story picking costs: short links in the editor's prompt, its share of the budget, the cost report."""
import json

from shorts import pipeline
from shorts.checks import check_picks
from shorts.llm import Usage, stage_name
from shorts.selection import (EDITOR_REPAIR_SHARE, EDITOR_TOOLS_SHARE, LONG_LINK, AgentEditor, LLMEditor,
                              SeenStore, ShortLinks)
from tests.test_agents import ScriptedLLM, _openai, _pick, _resp, _story
from tests.test_shadow import _stub_run

TRACKER = "https://link.mail.beehiiv.com/ss/c/u001." + "x" * 300  # what newsletter links look like
ARTICLE = "https://techcrunch.com/2026/10/01/chatgpt-can-now-virtually-try-on-clothes/"


def _letter(links=30):
    body = "\n".join(f"Story {i}: a lab shipped thing {i} (https://link.mail.beehiiv.com/ss/c/u001.{i:03d}"
                     f"{'y' * 300}). Read more (https://link.mail.beehiiv.com/ss/c/u001.{i:03d}{'y' * 300})."
                     for i in range(links))
    return _story("The Rundown", kind="newsletter", url="", source="The Rundown", body=body + f"\nAlso: ({ARTICLE})")


# --- short links -------------------------------------------------------------------------------

def test_long_links_become_numbered_and_come_back():
    links = ShortLinks()
    text = links.shorten(f"Try-on (link {TRACKER}). Again: {TRACKER}, and the article ({ARTICLE}).")
    assert text == f"Try-on (link link:1 on link.mail.beehiiv.com). Again: link:1 on link.mail.beehiiv.com, and the " \
                   f"article ({ARTICLE})."
    assert len(ARTICLE) <= LONG_LINK < len(TRACKER)
    for given in ("link:1", " link:1 on link.mail.beehiiv.com", "LINK:1"):
        assert links.expand(given) == TRACKER
    assert links.expand("link:2") == "link:2" and links.expand(ARTICLE) == ARTICLE and links.expand("") == ""


def test_the_editor_prompt_carries_far_fewer_link_characters(tmp_path):
    letter = _letter()
    editor = LLMEditor(None, 30)
    prompt = editor._prompt([letter], 8, SeenStore(tmp_path / "seen.json"))
    assert "beehiiv.com/ss" not in prompt and ARTICLE in prompt and "link:30 on link.mail.beehiiv.com" in prompt
    assert len(prompt) < len(letter.body) / 4
    assert len(editor.links.urls) == 30  # each link once, however often it appears


def _seen(tmp_path):
    return SeenStore(tmp_path / "seen.json")


def test_a_pick_given_as_a_short_link_gets_the_real_link_and_passes_the_checks(tmp_path):
    letter = _letter(3)
    real = f"https://link.mail.beehiiv.com/ss/c/u001.001{'y' * 300}"
    llm = ScriptedLLM({"stories": [_pick("Lab ships thing one", "link:2")]})
    editor = LLMEditor(llm, 30)
    [pick] = editor.pick([letter], 1, _seen(tmp_path))
    assert pick.url == real
    assert not [i for i in check_picks([pick], [letter], 1, set(), [], 30) if i.code == "url_not_in_sources"]


def test_the_agent_editor_expands_short_links_in_picks_alternates_and_repairs(tmp_path):
    letter, feed = _letter(3), _story("Nvidia unveils AI chip")
    first = {"stories": [_pick("Lab ships thing one", "link:2"), _pick("Lab ships thing one again", "link:2")]}
    fixed = {"stories": [_pick("Lab ships thing one", "link:2"), _pick("Nvidia unveils AI chip", "link:3")]}
    picks = AgentEditor(ScriptedLLM(first, fixed), 30).pick([letter, feed], 2, _seen(tmp_path))
    assert [p.url[:45] for p in picks] == ["https://link.mail.beehiiv.com/ss/c/u001.001yy",
                                           "https://link.mail.beehiiv.com/ss/c/u001.002yy"]


# --- the editor's share of the budget ----------------------------------------------------------

def _answer(*picks):
    return json.dumps({"stories": list(picks)})


def test_the_editor_stops_using_tools_once_it_has_spent_its_share(tmp_path):
    candidates = [_story("OpenAI ships GPT agent")]
    usage = Usage(run_cap_usd=0.60)
    search = ("search_candidates", {"query": "OpenAI"})
    # gpt-5: $1.25 per 1M input tokens, so 100k tokens is $0.125 and two such turns pass 30% of $0.60
    llm, fake = _openai(_resp(calls=[search], rid="r1", inp=100_000), _resp(calls=[search], rid="r2", inp=100_000),
                        _resp(text=_answer(_pick("OpenAI ships GPT agent", candidates[0].url)), rid="r3"),
                        usage=usage)
    picks = AgentEditor(llm, 30).pick(candidates, 1, _seen(tmp_path))
    assert [c.get("tool_choice") for c in fake.calls] == [None, "auto", "none"]
    assert picks[0].headline == "OpenAI ships GPT agent"
    assert usage.stage_usd("editor") > EDITOR_TOOLS_SHARE * 0.60


def test_no_repair_round_once_the_editor_has_spent_half_the_budget(tmp_path, caplog):
    candidates = [_story("OpenAI ships GPT agent"), _story("Nvidia unveils AI chip")]
    twice = _answer(_pick("OpenAI ships GPT agent", candidates[0].url), _pick("OpenAI ships a GPT agent",
                                                                              candidates[0].url))
    usage = Usage(run_cap_usd=0.60)
    llm, fake = _openai(_resp(text=twice, inp=250_000), usage=usage)  # $0.31, over half of $0.60
    picks = AgentEditor(llm, 30).pick(candidates, 2, _seen(tmp_path))
    assert len(fake.calls) == 1 and len(picks) == 2  # the duplicate is left to settle() and the top-up
    assert "no repair: the editor has spent $0.31" in caplog.text
    assert EDITOR_REPAIR_SHARE * 0.60 < 0.31


def test_without_a_cap_the_editor_has_no_share():
    assert AgentEditor(ScriptedLLM(), 30)._share(0.3) is None
    llm, _ = _openai(usage=Usage())
    assert AgentEditor(llm, 30)._share(0.3) is None
    llm, _ = _openai(usage=Usage(run_cap_usd=0.60))
    assert AgentEditor(llm, 30)._share(0.5) == 0.30


# --- where the money went ----------------------------------------------------------------------

def test_spend_is_reported_by_stage():
    usage = Usage()
    usage.add("editor", "gpt-5", 100_000, 2_000, cached_tokens=60_000)
    usage.add("editor-repair-1", "gpt-5", 20_000, 1_000)
    usage.add("research-3", "gpt-5", 8_000, 500)
    assert [stage_name(s) for s in ("editor-repair-1", "research-3", "writer-trim", "critic")] == \
        ["editor", "research", "writer", "critic"]
    rows = usage.by_stage()
    assert list(rows) == ["editor", "research"] and rows["editor"]["calls"] == 2
    assert rows["editor"]["input_tokens"] == 120_000 and rows["editor"]["cached_tokens"] == 60_000
    assert usage.stage_usd("editor") == rows["editor"]["usd"]
    assert usage.stage_lines()[0] == f"editor ${rows['editor']['usd']:.2f}: 2 calls, 120k tokens in (60k cached), 3.0k out"


def test_a_run_writes_the_stage_costs_to_cost_json_the_log_and_the_email(monkeypatch, tmp_path, caplog):
    from dataclasses import replace

    rig = _stub_run(monkeypatch, tmp_path)
    fake_llm = pipeline.build_llm

    def build_llm(cfg, usage):  # the fake LLM records no tokens: one priced call stands in for the editor
        usage.add("editor", "gpt-5", 100_000, 2_000)
        return fake_llm(cfg, usage)

    monkeypatch.setattr(pipeline, "build_llm", build_llm)
    caplog.set_level("INFO")
    video = pipeline.run(replace(rig.cfg, shadow=False), upload=False)
    cost = json.loads((video.parent / "cost.json").read_text())
    assert cost["by_stage"]["editor"] == {"usd": 0.145, "calls": 1, "input_tokens": 100_000, "cached_tokens": 0,
                                          "output_tokens": 2_000}
    assert "cost: editor $0.14: 1 call, 100k tokens in (0k cached), 2.0k out" in caplog.text
    assert "Cost: $0.14 (editor $0.14) ($" in rig.notes[-1][1]
    assert pipeline._stage_costs(Usage()) == ""
