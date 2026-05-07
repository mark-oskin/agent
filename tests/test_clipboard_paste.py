"""`/clipboard paste` loads clipboard without executing pasted text as REPL."""

from __future__ import annotations

from agentlib import session as session_mod
from agentlib.embedding import build_embedded_session


def test_clipboard_paste_sets_prefill_does_not_run_line(monkeypatch):
    monkeypatch.setattr(session_mod, "clipboard_read_text", lambda: "  phantom /quit  ")
    _, sess = build_embedded_session(verbose=0)

    before = len(sess.messages)
    r = sess.execute_line("/clipboard paste")
    after = len(sess.messages)

    assert after == before
    assert r.get("type") == "command"
    assert r.get("quit") is False
    assert r.get("prefill_prompt") == "  phantom /quit  "
    assert isinstance(r.get("output"), str) and "clipboard" in (r.get("output") or "").lower()


def test_clipboard_paste_prefill_with_emit(monkeypatch):
    monkeypatch.setattr(session_mod, "clipboard_read_text", lambda: "next line payload")
    calls: list[dict] = []

    _, sess = build_embedded_session(verbose=0)
    em = lambda ev: calls.append(ev)
    r = sess.execute_line("/clipboard paste", emit=em)

    assert r.get("type") == "command"
    assert r.get("quit") is False
    assert r.get("prefill_prompt") == "next line payload"
    out = r.get("output") or ""
    assert "17 characters" in out
    assert "edit" in out.lower()
