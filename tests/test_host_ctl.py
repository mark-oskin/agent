"""Tests for ``session.host_ctl`` and repl last-* snapshots without the TUI."""

from __future__ import annotations

from unittest.mock import patch

from agentlib.embedding import build_embedded_session


def test_host_ctl_last_answer_local_without_host():
    _, sess = build_embedded_session(verbose=0)
    sess.repl_last_user_query = "hello?"
    sess.repl_last_assistant_answer = "hi there"
    r = sess.host_ctl("last_answer")
    assert r["ok"] is True
    assert r["text"] == "hi there"


def test_host_ctl_last_question_local_without_host():
    _, sess = build_embedded_session(verbose=0)
    sess.repl_last_user_query = "ping"
    r = sess.host_ctl("last_question")
    assert r["ok"] is True
    assert r["text"] == "ping"


def test_host_ctl_empty_last_answer_hint():
    _, sess = build_embedded_session(verbose=0)
    r = sess.host_ctl("last_answer")
    assert r["ok"] is True
    assert "no last assistant answer" in r["text"].lower()


def test_host_ctl_list_requires_host():
    _, sess = build_embedded_session(verbose=0)
    r = sess.host_ctl("list_agents")
    assert r["ok"] is False


def test_execute_line_updates_repl_last_fields():
    _, sess = build_embedded_session(verbose=0)
    with patch.object(sess, "_execute_user_request", return_value=(True, "final reply")):
        sess.execute_line("What is 2+2?")
    assert sess.repl_last_user_query == "What is 2+2?"
    assert sess.repl_last_assistant_answer == "final reply"


def test_clear_resets_repl_last():
    _, sess = build_embedded_session(verbose=0)
    sess.repl_last_user_query = "x"
    sess.repl_last_assistant_answer = "y"
    sess.messages.append({"role": "user", "content": "keep-clear happy"})
    sess.execute_line("/clear")
    assert sess.repl_last_user_query is None
    assert sess.repl_last_assistant_answer is None
    assert sess.messages == []
