"""Unit tests for ``extensions/code.py`` verdict parsing and fork post-load lines."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from agentlib.settings import AgentSettings
from agentlib.tui_parse import parse_fork_background_command

_EXT = Path(__file__).resolve().parent.parent / "extensions" / "code.py"


@pytest.fixture(scope="module")
def code_ext():
    spec = importlib.util.spec_from_file_location("code_ext_under_test", _EXT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_pipeline_verdict_pass(code_ext):
    text = """Hello
---PIPELINE---
VERDICT: PASS
SUMMARY: all good
---END---
tail"""
    assert code_ext.parse_pipeline_verdict(text) == (True, "all good")


def test_parse_pipeline_verdict_fail(code_ext):
    text = """---PIPELINE---
VERDICT: FAIL
SUMMARY: broken
---END---"""
    assert code_ext.parse_pipeline_verdict(text) == (False, "broken")


def test_parse_pipeline_verdict_last_block_wins(code_ext):
    text = """---PIPELINE---
VERDICT: FAIL
SUMMARY: first
---END---
more
---PIPELINE---
VERDICT: PASS
SUMMARY: second
---END---"""
    assert code_ext.parse_pipeline_verdict(text) == (True, "second")


def test_parse_pipeline_verdict_none(code_ext):
    assert code_ext.parse_pipeline_verdict("no block here") is None
    assert code_ext.parse_pipeline_verdict("") is None


def test_post_load_fork_lines_parse_as_single_command(code_ext):
    """Regression: coder boot must parse (apostrophe in designer's; naive shlex.quote breaks on first ")."""
    for ln in code_ext._POST_LOAD_LINES:
        r = parse_fork_background_command(ln)
        assert r is not None, ln[:120]
        name, cmds = r
        assert name in ("designer", "coder", "reviewer", "tester")
        assert len(cmds) == 1, (name, len(cmds))


def test_pipeline_limits_reads_extensions_prefs(code_ext):
    class _S:
        settings: AgentSettings

    s = _S()
    s.settings = AgentSettings.defaults()
    s.settings.extensions_set_kv("code_pipeline", "code_test_max", "2")
    lim = code_ext._pipeline_limits(s)
    assert lim.code_test_max == 2
    assert lim.design_review_max >= 1


def test_parse_pipeline_verdict_preserves_multiline_summary(code_ext):
    text = """---PIPELINE---
VERDICT: PASS
SUMMARY: line one
  spaced  
line three
---END---"""
    v, s = code_ext.parse_pipeline_verdict(text)
    assert v is True
    assert "line one" in s
    assert "spaced" in s
    assert "line three" in s
    assert "\n" in s


def test_missing_verdict_nudge_requires_end(code_ext):
    n = code_ext._missing_verdict_nudge("some prior text")
    assert "---END---" in n
    assert "ORCHESTRATION" in n
    assert "some prior text" in n


def test_parse_or_retry_nudges_when_substantial_reply_has_no_verdict(code_ext):
    calls: list[str] = []

    def dl(role, cmd):
        calls.append(cmd)
        if len(calls) == 1:
            assert "ORCHESTRATION" not in cmd
            return {"type": "turn", "answer": "x" * 220 + "\n(no pipeline block)"}
        assert "ORCHESTRATION" in cmd
        assert "no pipeline block" in cmd
        return {
            "type": "turn",
            "answer": "---PIPELINE---\nVERDICT: PASS\nSUMMARY: fixed\n---END---\n",
        }

    class S:
        python_delegate_line = staticmethod(dl)
        settings = AgentSettings.defaults()

    lim = code_ext._pipeline_limits(S())
    pf = [0]
    ok, _ = code_ext._parse_or_retry(
        S(), "coder", lambda _prev: "BASE_TASK", parse_fails=pf, stage="unit", lim=lim
    )
    assert ok is True
    assert pf[0] == 0
    assert len(calls) == 2


def test_parse_or_retry_short_reply_increments_parse_fails(code_ext):
    calls: list[str] = []

    def dl(role, cmd):
        calls.append(cmd)
        return {"type": "turn", "answer": "short"}

    class S:
        python_delegate_line = staticmethod(dl)
        settings = AgentSettings.defaults()

    lim = code_ext._pipeline_limits(S())
    pf = [0]
    ok, _ = code_ext._parse_or_retry(
        S(), "designer", lambda _p: "ASK", parse_fails=pf, stage="unit", lim=lim
    )
    assert ok is False
    assert pf[0] == 5  # inner_round_max attempts, each short reply increments
    assert len(calls) == 5
