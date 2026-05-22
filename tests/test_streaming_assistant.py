"""Live assistant answer streaming during LLM generation."""

from __future__ import annotations

import json

from agentlib.llm import streaming as llm_streaming
from agentlib.llm.streaming import merge_stream_content, merge_visible_answer_text


def test_merge_stream_content_replaces_fresh_json_object():
    first = '{"action":"answer","answer":"partial list'
    restart = '{"action":"answer","answer":"Here are emails"}'
    assert llm_streaming.merge_stream_content(first, restart) == restart
    assert llm_streaming._stream_content_is_fresh_json_object(first, restart)


def test_merge_stream_content_keeps_cumulative_prefix():
    partial = '{"action":"answer","answer":"4'
    full = '{"action":"answer","answer":"4"}'
    assert llm_streaming.merge_stream_content(partial, full) == full
    assert not llm_streaming._stream_content_is_fresh_json_object(partial, full)


def test_stream_restart_does_not_bleed_glued_json_into_draft(monkeypatch):
    """Second JSON object glued onto the first must not appear inside Draft answer text."""
    emitted: list[dict] = []

    def cap(ev: dict) -> None:
        emitted.append(dict(ev))

    monkeypatch.setattr(llm_streaming, "sink_emit", cap)
    llm_streaming.reset_assistant_answer_streamed()

    part1 = '{"action":"answer","answer":"Email one'
    part2 = '"}'
    restart = '{"action":"answer","answer":"Email two"}'
    lines = [
        json.dumps({"message": {"content": part1}, "done": False}),
        json.dumps({"message": {"content": part2}, "done": False}),
        json.dumps({"message": {"content": restart}, "done": True}),
    ]
    llm_streaming.merge_stream_message_chunks(
        iter(lines),
        stream_user_visible=True,
        agent_stream_thinking_enabled=lambda: False,
    )
    answer_text = ""
    for e in emitted:
        if e.get("type") == "answer_reset":
            answer_text = ""
        elif e.get("type") == "answer":
            answer_text = e.get("text", "")
    assert answer_text == "Email two"
    assert '"action"' not in answer_text
    assert any(e.get("type") == "answer_reset" for e in emitted)


def test_stream_answer_rewrite_resets_draft(monkeypatch):
    """When the in-flight answer text is rewritten, Draft must not keep stale prefixes."""
    emitted: list[dict] = []

    def cap(ev: dict) -> None:
        emitted.append(dict(ev))

    monkeypatch.setattr(llm_streaming, "sink_emit", cap)
    llm_streaming.reset_assistant_answer_streamed()

    lines = [
        json.dumps(
            {
                "message": {
                    "content": '{"action":"answer","answer":": noreply@cs.w"}',
                },
                "done": False,
            }
        ),
        json.dumps(
            {
                "message": {
                    "content": '{"action":"answer","answer":"Here are your emails"}',
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
    answer_events = [e for e in emitted if e.get("type") == "answer"]
    assert answer_events
    assert all(e.get("full_snapshot") for e in answer_events)
    assert answer_events[-1]["text"] == "Here are your emails"
    assert any(e.get("type") == "answer_reset" for e in emitted)
    assert ": noreply" not in answer_events[-1]["text"]


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
    assert all(e.get("full_snapshot") for e in answer_events)
    assert answer_events[-1]["text"] == "Hello"
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
    answer_events = [e for e in emitted if e.get("type") == "answer"]
    assert answer_events[-1]["text"] == "4"


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
    answer_events = [e for e in emitted if e.get("type") == "answer"]
    assert answer_events[-1]["text"] == "Hi"
