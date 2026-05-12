"""Unit tests for ``extensions/code.py`` verdict parsing and fork post-load lines."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from agentlib.repl_extensions import ReplExtensionRegistry
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
            assert cmd.startswith("Always end with exactly one block")
            assert "────────\n\nBASE_TASK" in cmd
            assert "AUTOMATION — the orchestrator parses" in cmd
            return {"type": "turn", "answer": "x" * 220 + "\n(no pipeline block)"}
        assert "ORCHESTRATION" in cmd
        assert "no pipeline block" in cmd
        return {
            "type": "turn",
            "answer": "---PIPELINE---\nVERDICT: PASS\nSUMMARY: fixed\n---END---\n",
        }

    class S:
        python_fork_background_agent = staticmethod(lambda *_a, **_k: {"type": "fork", "ok": True})
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
        python_fork_background_agent = staticmethod(lambda *_a, **_k: {"type": "fork", "ok": True})
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
    for c in calls:
        assert "AUTOMATION — the orchestrator parses" in c


def test_parse_or_retry_nudges_when_first_reply_empty(code_ext):
    calls: list[str] = []

    def dl(role, cmd):
        calls.append(cmd)
        if len(calls) == 1:
            assert "ORCHESTRATION" not in cmd
            return {"type": "turn", "answer": ""}
        assert "ORCHESTRATION" in cmd
        assert "empty answer" in cmd or "step limit" in cmd
        return {
            "type": "turn",
            "answer": "---PIPELINE---\nVERDICT: PASS\nSUMMARY: ok\n---END---\n",
        }

    class S:
        python_fork_background_agent = staticmethod(lambda *_a, **_k: {"type": "fork", "ok": True})
        python_delegate_line = staticmethod(dl)
        settings = AgentSettings.defaults()

    lim = code_ext._pipeline_limits(S())
    pf = [0]
    ok, _ = code_ext._parse_or_retry(
        S(), "coder", lambda _prev: "BASE_TASK", parse_fails=pf, stage="unit", lim=lim
    )
    assert ok is True
    assert len(calls) == 2


def test_multilane_pipeline_available_requires_both_hooks(code_ext):
    class _ForkOnly:
        python_fork_background_agent = lambda *_a, **_k: None
        python_delegate_line = None

    class _DelOnly:
        python_fork_background_agent = None
        python_delegate_line = lambda *_a, **_k: None

    class _Both:
        python_fork_background_agent = lambda *_a, **_k: None
        python_delegate_line = lambda *_a, **_k: None

    assert not code_ext._multilane_pipeline_available(_ForkOnly())
    assert not code_ext._multilane_pipeline_available(_DelOnly())
    assert code_ext._multilane_pipeline_available(_Both())


def test_delegate_single_lane_uses_execute_line(code_ext):
    calls: list[str] = []

    def el(line: str, emit=None):
        calls.append(line)
        return {
            "type": "turn",
            "answer": "---PIPELINE---\nVERDICT: PASS\nSUMMARY: step\n---END---\n",
        }

    class S:
        python_fork_background_agent = None
        python_delegate_line = None
        execute_line = staticmethod(el)
        settings = AgentSettings.defaults()

    out = code_ext._delegate(S(), "designer", "  prompt  ")
    assert "PIPELINE" in out
    assert calls == ["prompt"]


def test_register_repl_post_load_empty_without_multilane(code_ext):
    class S:
        settings = AgentSettings.defaults()
        python_fork_background_agent = None
        python_delegate_line = None

        def _repl_register_extension_command(self, key, handler):
            self._cmd = (key, handler)

    s = S()
    reg = ReplExtensionRegistry(session=s, script_path=str(_EXT))
    post = code_ext.register_repl(s, reg)
    assert post == []


def test_register_repl_post_load_fork_lines_when_multilane(code_ext):
    class S:
        settings = AgentSettings.defaults()
        python_fork_background_agent = staticmethod(lambda *_a, **_k: {"type": "fork", "ok": True})
        python_delegate_line = staticmethod(lambda *_a, **_k: {"type": "turn", "answer": ""})

        def _repl_register_extension_command(self, key, handler):
            pass

    s = S()
    reg = ReplExtensionRegistry(session=s, script_path=str(_EXT))
    post = code_ext.register_repl(s, reg)
    assert isinstance(post, list)
    assert len(post) == len(code_ext._POST_LOAD_LINES)


def test_run_pipeline_single_lane_happy_path(code_ext):
    verdict = "---PIPELINE---\nVERDICT: PASS\nSUMMARY: ok\n---END---\n"
    calls: list[str] = []

    def el(line: str, emit=None):
        calls.append(line)
        return {"type": "turn", "answer": verdict}

    class S:
        settings = AgentSettings.defaults()
        session_cwd = "/tmp"
        python_fork_background_agent = None
        python_delegate_line = None
        execute_line = staticmethod(el)

    msg = code_ext._run_pipeline(S(), "add a small feature")
    assert "Pipeline complete" in msg
    assert len(calls) == 4


def test_parse_code_rest_empty(code_ext):
    f, ask = code_ext._parse_code_rest("")
    assert ask == ""
    assert not any(f)


def test_parse_code_rest_flags_and_ask(code_ext):
    f, ask = code_ext._parse_code_rest("  --skip_design   --skip_test  fix login  ")
    assert f.skip_design and f.skip_test and not f.skip_review
    assert ask == "fix login"


def test_parse_code_rest_all_flags_order_independent_for_ask(code_ext):
    f, ask = code_ext._parse_code_rest("--skip_review --skip_design do thing")
    assert f.skip_design and f.skip_review and not f.skip_test
    assert ask == "do thing"


def test_parse_code_rest_unknown_token_stops_flags(code_ext):
    f, ask = code_ext._parse_code_rest("--skip_design --not_a_flag rest")
    assert f.skip_design
    assert ask == "--not_a_flag rest"


def test_run_pipeline_single_lane_skip_all_but_coder(code_ext):
    verdict = "---PIPELINE---\nVERDICT: PASS\nSUMMARY: ok\n---END---\n"
    calls: list[str] = []

    def el(line: str, emit=None):
        calls.append(line)
        return {"type": "turn", "answer": verdict}

    class S:
        settings = AgentSettings.defaults()
        session_cwd = "/tmp"
        python_fork_background_agent = None
        python_delegate_line = None
        execute_line = staticmethod(el)

    msg = code_ext._run_pipeline(S(), "--skip_design --skip_review --skip_test only coder runs")
    assert "Pipeline complete" in msg
    assert len(calls) == 1
