"""Tests for agentlib.sink incremental emit routing."""

from __future__ import annotations

from agentlib.sink import emit_sink_scope, sink_emit, sink_print_compat


def test_emit_sink_scope_streams_before_scope_exit():
    events: list[dict] = []

    def emit(ev: dict) -> None:
        events.append(dict(ev))

    with emit_sink_scope(emit):
        sink_emit({"type": "thinking", "text": "a", "end": "", "partial": True})
        sink_emit({"type": "thinking", "text": "b", "end": "", "partial": True})

    assert events == [
        {"type": "thinking", "text": "a", "end": "", "partial": True},
        {"type": "thinking", "text": "b", "end": "", "partial": True},
    ]


def test_emit_sink_scope_cleared_after_exit():
    seen: list[str] = []

    def emit(ev: dict) -> None:
        seen.append("x")

    with emit_sink_scope(emit):
        sink_emit({"type": "output", "text": "during"})
    sink_emit({"type": "output", "text": "after"})
    assert len(seen) == 1
