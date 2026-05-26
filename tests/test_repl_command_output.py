"""Slash-command feedback is returned in ``output`` once (no duplicate sink + host print)."""

from __future__ import annotations

from agentlib.embedding import build_embedded_session


def test_clear_with_emit_does_not_duplicate_sink_output():
    calls: list[dict] = []
    _, sess = build_embedded_session(verbose=0)
    r = sess.execute_line("/clear", emit=lambda ev: calls.append(ev))
    msg = "Context cleared (including stored skill for /skill reuse)."
    sink_hits = [
        c
        for c in calls
        if c.get("type") == "output" and msg in (c.get("text") or "")
    ]
    assert sink_hits == []
    assert r.get("type") == "command"
    assert (r.get("output") or "").strip() == msg


def test_clear_without_emit_returns_output_only():
    _, sess = build_embedded_session(verbose=0)
    r = sess.execute_line("/clear")
    assert "Context cleared" in (r.get("output") or "")
