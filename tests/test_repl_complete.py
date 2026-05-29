"""Tests for REPL tab completion and command registry dispatch."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

from agentlib.repl.command_registry import (
    ReplCompletionContext,
    dispatch_repl_command,
    find_spec_for_line,
    format_repl_help,
    iter_visible_command_names,
)
from agentlib.repl.complete import apply_repl_completion, complete_repl_candidates
from tests.harness import build_test_session, run_session_lines

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_complete_top_level_commands(monkeypatch):
    _app, session = build_test_session(monkeypatch, verbose=0)
    names = complete_repl_candidates(session, "/se", 3)
    assert "/send" in names or "/set" in names
    assert all(n.startswith("/") for n in names)


def test_complete_context_subcommands(monkeypatch):
    _app, session = build_test_session(monkeypatch, verbose=0)
    names = complete_repl_candidates(session, "/context ", 9)
    assert "load" in names
    assert "save" in names


def test_apply_completion_single_match_adds_space():
    result = apply_repl_completion("/he", 3, ["/help"])
    assert result.line == "/help "
    assert result.cursor == len("/help ")
    assert not result.list_candidates


def test_apply_completion_lists_when_ambiguous():
    result = apply_repl_completion("/set model ", 11, ["alpha:1", "beta:2"])
    assert result.line == "/set model "
    assert result.cursor == 11
    assert result.list_candidates == ("alpha:1", "beta:2")


def test_apply_completion_extends_common_prefix():
    result = apply_repl_completion(
        "/set model gemma",
        16,
        ["gemma4:26b", "gemma4:31b"],
    )
    assert result.line == "/set model gemma4:"
    assert not result.list_candidates

def test_fork_background_matches_before_fork(monkeypatch):
    _app, session = build_test_session(monkeypatch, verbose=0)
    session.python_fork_agent = lambda *a, **k: {"ok": True}
    spec = find_spec_for_line("/fork_background Worker", session)
    assert spec is not None
    assert spec.name == "/fork_background"
    spec2 = find_spec_for_line("/fork Worker", session)
    assert spec2 is not None
    assert spec2.name == "/fork"


def test_dispatch_quit_and_clear(monkeypatch):
    _app, session = build_test_session(monkeypatch, verbose=0)
    session.messages.append({"role": "user", "content": "hi"})
    res = dispatch_repl_command(session, "/clear")
    assert res is not None
    assert not res.quit
    assert session.messages == []
    res2 = dispatch_repl_command(session, "/quit")
    assert res2 is not None
    assert res2.quit


def test_help_includes_core_commands(monkeypatch):
    _app, session = build_test_session(monkeypatch, verbose=0)
    text = format_repl_help(session)
    assert "/quit" in text
    assert "/context load" in text
    assert "/call_python" in text


def test_list_command_exact_only(monkeypatch):
    _app, session = build_test_session(monkeypatch, verbose=0)
    session.python_host_command = lambda _p: {}
    assert find_spec_for_line("/list", session) is not None
    assert find_spec_for_line("/list extra", session) is None


def test_extension_command_completes_after_load(tmp_path, monkeypatch):
    simple = FIXTURES / "repl_ext_simple.py"
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, [f"/load {simple}", "/quit"])
    names = complete_repl_candidates(session, "/ext", 4)
    assert "/ext_ok" in names


def test_complete_switch_agent_labels(monkeypatch):
    _app, session = build_test_session(monkeypatch, verbose=0)
    session.python_host_command = lambda _p: {}
    ctx = ReplCompletionContext(agent_labels=("Planner", "Coder"))
    names = complete_repl_candidates(session, "/switch Cod", 11, ctx=ctx)
    assert "Coder" in names


def test_visible_command_names_include_aliases(monkeypatch):
    _app, session = build_test_session(monkeypatch, verbose=0)
    names = iter_visible_command_names(session)
    assert "/tokens" in names
    assert "/settings" in names


def test_tui_tab_complete_prompt():
    import asyncio

    async def run():
        from agent_tui import AgentTuiApp
        from textual.widgets import TextArea

        async with AgentTuiApp(verbose=0, agent_specs=["A"]).run_test(size=(80, 24)) as pilot:
            pr = pilot.app.query_one("#prompt", TextArea)
            pr.text = "/he"
            pr.move_cursor((0, 3))
            pilot.app.repl_tab_complete_prompt(pr)
            assert pr.text == "/help "
            assert pr.cursor_location == (0, 6)

    asyncio.run(run())


def test_tui_tab_complete_lists_ambiguous():
    import asyncio

    async def run():
        from agent_tui import AgentTuiApp
        from textual.actions import SkipAction
        from textual.widgets import TextArea

        async with AgentTuiApp(verbose=0, agent_specs=["A"]).run_test(size=(80, 24)) as pilot:
            app = pilot.app
            pr = app.query_one("#prompt", TextArea)
            lane = app._active_lane
            app._sessions[lane].repl_completion_ollama_models = (
                "alpha:1",
                "beta:2",
            )
            pr.text = "/set model "
            pr.move_cursor((0, 11))
            try:
                app.repl_tab_complete_prompt(pr)
            except SkipAction:
                pass
            assert pr.text == "/set model "
            log = app._chat_logs[lane]
            rendered = "\n".join(log._plain_lines)
            assert "alpha:1" in rendered
            assert "beta:2" in rendered

    asyncio.run(run())


def test_tui_ctrl_c_clears_prompt():
    import asyncio

    async def run():
        from agent_tui import AgentTuiApp
        from textual.widgets import TextArea

        async with AgentTuiApp(verbose=0, agent_specs=["A"]).run_test(size=(80, 24)) as pilot:
            app = pilot.app
            pr = app.query_one("#prompt", TextArea)
            pr.focus()
            pr.text = "/set model foo"
            app._prompt_hist_idx[app._active_lane] = 0
            app.action_interrupt_prompt()
            assert pr.text == ""
            assert pr.cursor_location == (0, 0)
            assert app._prompt_hist_idx[app._active_lane] is None

    asyncio.run(run())


def test_complete_set_model_ollama_names(monkeypatch):
    _app, session = build_test_session(monkeypatch, verbose=0)
    session.repl_completion_ollama_models = ("llama3.2:latest", "qwen2.5-coder:latest")
    names = complete_repl_candidates(session, "/set model ll", 13)
    assert names == ["llama3.2:latest"]


def test_complete_show_model_ollama_names(monkeypatch):
    _app, session = build_test_session(monkeypatch, verbose=0)
    session.repl_completion_ollama_models = ("llama3.2:latest", "qwen2.5-coder:latest")
    names = complete_repl_candidates(session, "/show model qwen", 16)
    assert names == ["qwen2.5-coder:latest"]


def test_probe_ollama_model_names_empty_on_failure(monkeypatch):
    from agentlib.repl.ollama_models import probe_ollama_model_names

    def boom(*_a, **_k):
        raise OSError("nope")

    assert probe_ollama_model_names("http://127.0.0.1:1", http_get=boom, timeout=0.1) == ()

