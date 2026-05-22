"""Live assistant answer streaming during LLM generation."""

from __future__ import annotations

import json

from agentlib.llm import streaming as llm_streaming
from agentlib.llm.streaming import merge_stream_content, merge_visible_answer_text


def test_merge_stream_content_cumulative_only():
    assert merge_stream_content("2 + 2 = ", "2 + 2 = 4") == "2 + 2 = 4"
    assert merge_stream_content("2 + 2 = 4", "2 + 2 = 4") == "2 + 2 = 4"
    assert merge_stream_content("Hello", "Hel") == "Hello"
    # Must not drop JSON substrings that repeat inside partial tool-call bodies.
    acc = '{"action":"tool_call","tool":"search_web"'
    chunk = "tool_call"
    assert merge_stream_content(acc, chunk) == acc + chunk


def test_merge_visible_answer_text_overlapping_prefix():
    assert merge_visible_answer_text("Donald Trump.", "The current President") == (
        "Donald Trump.The current President"
    )
    assert merge_visible_answer_text("Hello", "lo world") == "Hello world"
    assert merge_visible_answer_text("2 + 2 equals 4. ", "+ 2 equals 4.") == "2 + 2 equals 4. "
    assert merge_visible_answer_text("2 + 2 equals 4.", " equals 4.") == "2 + 2 equals 4."
    assert merge_visible_answer_text("2 +", " 2") == "2 + 2"


def test_merge_stream_emits_partial_answer_events(monkeypatch):
    emitted: list[dict] = []

    def cap(ev: dict) -> None:
        emitted.append(dict(ev))

    monkeypatch.setattr(llm_streaming, "sink_emit", cap)
    llm_streaming.reset_assistant_answer_streamed()

    lines = [
        json.dumps(
            {
                "message": {"content": '{"action":"answer","answer":"Hel'},
                "done": False,
            }
        ),
        json.dumps(
            {
                "message": {"content": 'lo"}'},
                "done": True,
            }
        ),
    ]
    msg, _, streamed = llm_streaming.merge_stream_message_chunks(
        iter(lines),
        stream_user_visible=True,
        agent_stream_thinking_enabled=lambda: False,
    )
    assert msg["content"] == '{"action":"answer","answer":"Hello"}'
    assert streamed is True
    answer_events = [e for e in emitted if e.get("type") == "answer"]
    assert answer_events
    assert all(e.get("partial") for e in answer_events)
    assert "".join(e.get("text", "") for e in answer_events) == "Hello"
    assert llm_streaming.assistant_answer_was_streamed()


def test_merge_stream_tool_call_buffers_no_answer_stream(monkeypatch):
    emitted: list[dict] = []

    def cap(ev: dict) -> None:
        emitted.append(dict(ev))

    monkeypatch.setattr(llm_streaming, "sink_emit", cap)
    llm_streaming.reset_assistant_answer_streamed()

    lines = [
        json.dumps(
            {
                "message": {
                    "content": '{"action":"tool_call","tool":"search_web","parameters":{"query":"x"}}'
                },
                "done": True,
            }
        ),
    ]
    llm_streaming.merge_stream_message_chunks(
        iter(lines),
        stream_user_visible=True,
        agent_stream_thinking_enabled=lambda: False,
    )
    assert not any(e.get("type") == "answer" for e in emitted)
    assert any(e.get("type") == "progress" for e in emitted)


def test_merge_stream_cumulative_chunks_do_not_double_answer(monkeypatch):
    """Backends that resend full cumulative JSON must not duplicate answer text."""
    emitted: list[dict] = []

    def cap(ev: dict) -> None:
        emitted.append(dict(ev))

    monkeypatch.setattr(llm_streaming, "sink_emit", cap)
    llm_streaming.reset_assistant_answer_streamed()

    partial = '{"action":"answer","answer":"4'
    full = '{"action":"answer","answer":"4"}'
    lines = [
        json.dumps({"message": {"content": partial}, "done": False}),
        json.dumps({"message": {"content": full}, "done": False}),
        json.dumps({"message": {"content": full}, "done": True}),
    ]
    msg, _, streamed = llm_streaming.merge_stream_message_chunks(
        iter(lines),
        stream_user_visible=True,
        agent_stream_thinking_enabled=lambda: False,
    )
    assert msg["content"] == full
    assert streamed is True
    assert "".join(e.get("text", "") for e in emitted if e.get("type") == "answer") == "4"


def test_merge_hosted_stream_accumulates_answer(monkeypatch):
    emitted: list[dict] = []

    def cap(ev: dict) -> None:
        emitted.append(dict(ev))

    monkeypatch.setattr(llm_streaming, "sink_emit", cap)
    llm_streaming.reset_assistant_answer_streamed()

    content = '{"action":"answer","answer":"Hi"}'
    sse = []
    for ch in content:
        sse.append(
            {
                "choices": [
                    {"delta": {"content": ch}, "finish_reason": None},
                ]
            }
        )
    sse.append({"_sse_done": True})
    msg, usage, streamed = llm_streaming.merge_hosted_stream_chunks(
        iter(sse),
        stream_user_visible=True,
        agent_stream_thinking_enabled=lambda: False,
    )
    assert msg["content"] == content
    assert usage is None
    assert streamed is True
    assert "".join(e.get("text", "") for e in emitted if e.get("type") == "answer") == "Hi"
