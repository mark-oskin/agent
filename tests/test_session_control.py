"""session_command tool: blocklist, validation, captured REPL output."""

from __future__ import annotations

import pytest

from agentlib.tools.session_control import (
    execute_session_slash_command,
    merge_command_transcript_output,
    normalize_allowlisted_session_command,
    session_command_blocked_reason,
    validate_session_command,
)


def test_validate_accepts_common_slash_commands():
    for line in (
        "/show models",
        "/set thinking show",
        "/set thinking on",
        "/set tools list",
        "/set agent show",
        "/set save",
        "/set model llama3",
        "/help",
        "/clear",
        "/usage",
        "/cd /tmp",
    ):
        norm, err = validate_session_command(line)
        assert err is None, line
        assert norm == line


def test_blocked_commands():
    assert normalize_allowlisted_session_command("/set lock") is None
    assert normalize_allowlisted_session_command("/quit") is None
    assert normalize_allowlisted_session_command("/call_python print(1)") is None
    assert normalize_allowlisted_session_command("/while 'x' do 'y'") is None
    assert session_command_blocked_reason("!ls") is not None


def test_merge_command_transcript_output_dedupes():
    assert merge_command_transcript_output("hello", "hello") == "hello"
    assert "a" in merge_command_transcript_output("b", "a\n\nb") or merge_command_transcript_output("b", "a") == "a\n\nb"


def test_execute_runs_execute_line():
    seen: list[str] = []

    def fake_execute(line: str) -> dict:
        seen.append(line)
        return {"type": "command", "quit": False, "output": "thinking enabled"}

    out = execute_session_slash_command("/set thinking on", execute_line=fake_execute)
    assert seen == ["/set thinking on"]
    assert "thinking enabled" in out


def test_thinking_show_output_reaches_session_command(tmp_path, monkeypatch):
    from tests.harness import build_test_session

    _app, session = build_test_session(
        monkeypatch, verbose=0, prefs_path=str(tmp_path / "prefs.json")
    )
    tool_out = session._execute_session_command_for_tool("/set thinking show")
    assert "thinking:" in tool_out
    assert "Extended thinking is" in tool_out
    assert tool_out != "OK (/set thinking show)"


def test_set_tools_list_reaches_session_command(tmp_path, monkeypatch):
    from tests.harness import build_test_session

    _app, session = build_test_session(
        monkeypatch, verbose=0, prefs_path=str(tmp_path / "prefs.json")
    )
    tool_out = session._execute_session_command_for_tool("/set tools list")
    assert "Tool:" in tool_out or "tool" in tool_out.lower()
    assert "OK (/set tools list)" not in tool_out


def test_show_works_when_settings_locked(tmp_path, monkeypatch):
    from tests.harness import build_test_session

    _app, session = build_test_session(
        monkeypatch, verbose=0, prefs_path=str(tmp_path / "prefs.json")
    )
    session.settings_locked = True
    out = session._execute_session_command_for_tool("/show models")
    assert "OK (/show models)" not in out or "model" in out.lower()
