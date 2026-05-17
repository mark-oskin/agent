"""Per-session /set lock — blocks settings and MCP mutations."""

from __future__ import annotations

import pytest

from agentlib.embedding import fork_embedded_session
from tests.harness import build_test_session


def test_set_lock_blocks_mutation_allows_show(monkeypatch, capsys, tmp_path):
    _app, session = build_test_session(monkeypatch, verbose=0, prefs_path=str(tmp_path / "agent.json"))
    session.execute_line("/set lock")
    capsys.readouterr()

    session.execute_line("/set verbose on")
    out = capsys.readouterr().out
    assert "locked" in out.lower()
    assert session.verbose == 0

    session.execute_line("/set ollama show")
    out = capsys.readouterr().out
    assert "locked" not in out.lower() or "ollama" in out.lower()

    session.execute_line("/mcp enable")
    out = capsys.readouterr().out
    assert "locked" in out.lower()

    session.execute_line("/mcp list")
    out = capsys.readouterr().out
    assert "locked" not in out.lower() or "mcp_enabled" in out


def test_set_unlock_is_not_available(monkeypatch, capsys, tmp_path):
    _app, session = build_test_session(monkeypatch, verbose=0, prefs_path=str(tmp_path / "agent.json"))
    session.execute_line("/set lock")
    capsys.readouterr()
    session.execute_line("/set unlock")
    out = capsys.readouterr().out
    assert "no /set unlock" in out.lower() or "permanent" in out.lower()
    assert session.settings_locked
    session.execute_line("/set verbose on")
    capsys.readouterr()
    assert session.verbose == 0


def test_fork_copies_settings_lock(monkeypatch, tmp_path):
    app, parent = build_test_session(monkeypatch, verbose=0, prefs_path=str(tmp_path / "agent.json"))
    parent.settings_locked = True
    child = fork_embedded_session(parent, app=app)
    assert child.settings_locked
