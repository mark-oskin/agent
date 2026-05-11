"""Tests for ``/load`` / ``/unload`` REPL extension modules."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

from tests.harness import build_test_session, run_session_lines

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_repl_load_unload_extension_commands(tmp_path, monkeypatch):
    simple = FIXTURES / "repl_ext_simple.py"
    lines = [
        f"/load {simple}",
        "/ext_ok hello",
        "/unload",
        "/ext_ok should_fail",
        "/quit",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert "ext_ok:hello" in out
    assert "Unknown command '/ext_ok'" in out


def test_repl_load_invoke_call_python_bridge(tmp_path, monkeypatch):
    bridge = FIXTURES / "repl_ext_bridge.py"
    lines = [
        f"/load {bridge}",
        "/pingx hi there",
        "/unload",
        "/quit",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert "pong:hi there" in out


def test_repl_unload_when_empty(monkeypatch):
    lines = ["/unload", "/quit"]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    assert "No REPL extensions loaded" in buf.getvalue()


def test_repl_extension_register_help_shown_in_slash_help(monkeypatch):
    simple = FIXTURES / "repl_ext_simple.py"
    lines = [
        f"/load {simple}",
        "/help",
        "/unload",
        "/help",
        "/quit",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        _app, session = build_test_session(monkeypatch, verbose=0)
        run_session_lines(session, lines)
    out = buf.getvalue()
    assert "Loaded extensions (/load):" in out
    assert "/ext_ok" in out
