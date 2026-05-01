"""Tests for ``/send`` and ``send()`` forwarding via enqueue / delegate hooks."""

from __future__ import annotations

from agentlib.embedding import build_embedded_session


def test_send_prefers_enqueue_when_configured():
    calls: list[tuple[str, str, str]] = []

    def eq(agent: str, cmd: str) -> dict:
        calls.append(("enqueue", agent, cmd))
        return {"ok": True, "queued": False, "label": agent}

    def dl(agent: str, cmd: str) -> dict:
        calls.append(("delegate", agent, cmd))
        return {"type": "command", "quit": False, "output": ""}

    _, sess = build_embedded_session(verbose=0)
    sess.python_enqueue_line = eq
    sess.python_delegate_line = dl
    sess.execute_line("/send Bob /help")
    assert calls == [("enqueue", "Bob", "/help")]


def test_send_delegates_to_hook():
    calls: list[tuple[str, str]] = []

    def dl(agent: str, cmd: str) -> dict:
        calls.append((agent, cmd))
        return {"type": "command", "quit": False, "output": ""}

    _, sess = build_embedded_session(verbose=0)
    sess.python_delegate_line = dl
    sess.execute_line("/send Bob /help")
    assert calls == [("Bob", "/help")]


def test_send_preserves_multiword_command():
    calls: list[tuple[str, str]] = []

    def dl(agent: str, cmd: str) -> dict:
        calls.append((agent, cmd))
        return {"type": "command", "quit": False, "output": ""}

    _, sess = build_embedded_session(verbose=0)
    sess.python_delegate_line = dl
    sess.execute_line('/send Bob hello world')
    assert calls == [("Bob", "hello world")]


def test_send_requires_delegate():
    _, sess = build_embedded_session(verbose=0)
    r = sess.execute_line("/send Bob hi")
    assert r["type"] == "command"
