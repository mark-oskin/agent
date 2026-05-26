"""CLI Draft/Final answer display via sink."""

from __future__ import annotations

from agentlib.llm import streaming as llm_streaming
from agentlib.sink import (
    DRAFT_LABEL,
    FINAL_LABEL,
    emit_sink_scope,
    print_turn_final_answer,
    reset_cli_answer_display,
    set_sink_show_draft,
    sink_emit,
    suppress_final_after_streamed_answer,
)


def test_cli_draft_then_final(capsys):
    llm_streaming.reset_assistant_answer_streamed()
    reset_cli_answer_display()
    set_sink_show_draft(True)
    sink_emit({"type": "answer", "text": "2 + 2 = ", "partial": True, "end": "", "flush": True})
    sink_emit({"type": "answer", "text": "4", "partial": True, "end": "", "flush": True})
    sink_emit({"type": "answer", "text": "2 + 2 = 4", "partial": True, "end": "", "flush": True})
    print_turn_final_answer("2 + 2 = 4")
    out = capsys.readouterr().out
    assert out.count(DRAFT_LABEL) == 1
    assert out.count(FINAL_LABEL) == 1
    assert "2 + 2 = 4" in out
    assert "2 + 2 = 42 + 2 = 4" not in out


def test_cli_stream_without_draft_label_when_show_draft_off(capsys):
    llm_streaming.reset_assistant_answer_streamed()
    reset_cli_answer_display()
    set_sink_show_draft(False)
    sink_emit({"type": "answer", "text": "hello", "partial": True, "end": "", "flush": True})
    sink_emit({"type": "answer", "text": " world", "partial": True, "end": "", "flush": True})
    assert suppress_final_after_streamed_answer("hello world")
    print_turn_final_answer("hello world")
    out = capsys.readouterr().out
    assert DRAFT_LABEL not in out
    assert FINAL_LABEL not in out
    assert out == "hello world"


def test_cli_stream_with_draft_still_prints_final(capsys):
    llm_streaming.reset_assistant_answer_streamed()
    reset_cli_answer_display()
    set_sink_show_draft(True)
    sink_emit({"type": "answer", "text": "hi", "partial": True, "end": "", "flush": True})
    assert not suppress_final_after_streamed_answer("hi")
    print_turn_final_answer("hi")
    out = capsys.readouterr().out
    assert DRAFT_LABEL in out
    assert f"{FINAL_LABEL}\nhi\n" in out


def test_cli_final_only_when_no_stream(capsys):
    llm_streaming.reset_assistant_answer_streamed()
    reset_cli_answer_display()
    print_turn_final_answer("hello")
    out = capsys.readouterr().out
    assert DRAFT_LABEL not in out
    assert f"{FINAL_LABEL}\nhello\n" in out


def test_emit_sink_forwards_final_answer_event():
    seen: list[dict] = []

    def cap(ev: dict) -> None:
        seen.append(dict(ev))

    with emit_sink_scope(cap):
        print_turn_final_answer("done")
    assert len(seen) == 1
    assert seen[0]["type"] == "final_answer"
    assert seen[0]["text"] == "done"
