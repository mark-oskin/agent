"""Provider-shaped transcript helpers."""

from agentlib.transcript_shape import (
    assistant_transcript_message,
    first_tool_call_id,
    tool_transcript_message,
)


def test_assistant_transcript_preserves_tool_calls():
    raw = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "function": {"name": "search_web", "arguments": '{"query":"x"}'},
            }
        ],
    }
    msg = assistant_transcript_message(raw, fallback_content='{"action":"tool_call"}', use_provider_shape=True)
    assert msg["role"] == "assistant"
    assert msg["tool_calls"]
    assert msg["content"] == ""


def test_assistant_transcript_json_fallback_when_no_tool_calls():
    msg = assistant_transcript_message(None, fallback_content="plain answer", use_provider_shape=True)
    assert msg == {"role": "assistant", "content": "plain answer"}


def test_tool_transcript_message_shape():
    msg = tool_transcript_message("search_web", "results", tool_call_id="call_1")
    assert msg["role"] == "tool"
    assert msg["tool_name"] == "search_web"
    assert msg["tool_call_id"] == "call_1"
    assert msg["content"] == "results"


def test_normalize_transcript_preserves_tool_fields():
    from agentlib.prompts import normalize_transcript_messages

    loaded = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "content": "out", "tool_name": "grep", "tool_call_id": "c1"},
    ]
    out = normalize_transcript_messages(loaded)
    assert out[0]["tool_calls"]
    assert out[1]["tool_name"] == "grep"
