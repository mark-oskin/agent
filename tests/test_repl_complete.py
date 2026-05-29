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
    line, cur = apply_repl_completion("/he", 3, ["/help"])
    assert line == "/help "
    assert cur == len("/help ")


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
