"""Tests for ``/run_command`` and ``!`` shell shorthand on ``AgentSession``."""

import os

from agentlib.embedding import build_embedded_session


def test_run_command_help():
    _, sess = build_embedded_session(verbose=0)
    r = sess.execute_line("/run_command help")
    assert r["type"] == "command"
    assert not r.get("quit")


def test_run_command_invokes_backend(monkeypatch, capsys):
    from agentlib.tools import builtins as tb

    calls = []

    def stub(cmd, cwd=None):
        calls.append((cmd, cwd))
        return "STDOUT:\nstubbed\nSTDERR:\n"

    monkeypatch.setattr(tb, "run_command", stub)
    _, sess = build_embedded_session(verbose=0)
    sess.execute_line("/run_command echo hello world")
    assert calls == [("echo hello world", sess.session_cwd)]
    assert "stubbed" in capsys.readouterr().out


def test_bang_shorthand(monkeypatch, capsys):
    from agentlib.tools import builtins as tb

    calls = []

    def stub(cmd, cwd=None):
        calls.append((cmd, cwd))
        return "STDOUT:\nok\nSTDERR:\n"

    monkeypatch.setattr(tb, "run_command", stub)
    _, sess = build_embedded_session(verbose=0)
    sess.execute_line("! pwd")
    assert calls == []
    out = capsys.readouterr().out
    assert sess.session_cwd in out


def test_bang_empty_usage(capsys):
    _, sess = build_embedded_session(verbose=0)
    sess.execute_line("!")
    out = capsys.readouterr().out.lower()
    assert "usage" in out or "/run_command" in out


def test_run_command_via_emit(monkeypatch):
    from agentlib.tools import builtins as tb

    events = []

    def stub(cmd, cwd=None):
        return f"STDOUT:\n{cmd}\nSTDERR:\n"

    monkeypatch.setattr(tb, "run_command", stub)
    _, sess = build_embedded_session(verbose=0)

    def emit(ev):
        events.append(ev)

    r = sess.execute_line("! x-test-cmd", emit=emit)
    assert r["type"] == "command"
    assert any("x-test-cmd" in str(ev) for ev in events)
