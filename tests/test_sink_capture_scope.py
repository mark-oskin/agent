from __future__ import annotations

from agentlib.sink import emit_sink_scope, sink_capture_scope, sink_print_compat


def test_sink_capture_scope_collects_without_parent():
    with sink_capture_scope(tee=False) as buf:
        sink_print_compat("line one")
        sink_print_compat("line two")
    assert "".join(buf).strip() == "line one\nline two"


def test_sink_capture_scope_tees_to_parent():
    parent: list[dict] = []

    def parent_emit(ev: dict) -> None:
        parent.append(ev)

    with emit_sink_scope(parent_emit):
        with sink_capture_scope(tee=True) as buf:
            sink_print_compat("hello")
    assert any(e.get("text") == "hello" for e in parent)
    assert "hello" in "".join(buf)
