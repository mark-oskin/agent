"""--load_context / --save_context JSON helpers and argv wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentlib.context.io import load_context_messages, parse_context_messages_data, save_context_bundle


def test_parse_context_messages_bare_list():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert parse_context_messages_data(msgs) == msgs


def test_parse_context_messages_wrapped():
    data = {"messages": [{"role": "user", "content": "a"}], "extra": 1}
    assert parse_context_messages_data(data) == [{"role": "user", "content": "a"}]


def test_parse_context_invalid_role():
    with pytest.raises(ValueError, match="invalid role"):
        parse_context_messages_data([{"role": "nope", "content": "x"}])


def test_save_and_reload_roundtrip(tmp_path: Path):
    p = tmp_path / "ctx.json"
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": '{"action":"answer","answer":"one"}'},
    ]
    save_context_bundle(str(p), messages, "follow-up", "two", True)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["user_query"] == "follow-up"
    assert data["final_answer"] == "two"
    assert data["answered"] is True
    assert len(data["messages"]) == 2
    loaded = load_context_messages(str(p))
    assert loaded == messages


def test_save_context_via_run_main(tmp_path: Path, monkeypatch):
    from tests.harness import run_main

    out_path = tmp_path / "saved.json"
    fin = json.dumps(
        {
            "action": "answer",
            "answer": "done",
            "next_action": "finalize",
            "rationale": "ok",
        }
    )
    run_main(
        monkeypatch,
        ["--save_context", str(out_path), "hello"],
        [
            json.dumps({"action": "answer", "answer": "first"}),
            fin,
            fin,
        ],
    )
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["user_query"] == "hello"
    assert data["final_answer"] == "done"
    assert data["answered"] is True
    assert any("hello" in m.get("content", "") for m in data["messages"] if m["role"] == "user")


def test_parse_context_messages_native_tool_shape():
    msgs = [
        {"role": "user", "content": "search iran"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "function": {"name": "search_web", "arguments": "{}"}}],
        },
        {
            "role": "tool",
            "content": "results…",
            "tool_name": "search_web",
            "tool_call_id": "call_1",
        },
        {"role": "assistant", "content": "Summary here."},
    ]
    loaded = parse_context_messages_data(msgs)
    assert loaded == msgs


def test_save_and_reload_native_tool_roundtrip(tmp_path: Path):
    from agentlib.prompts import normalize_transcript_messages

    p = tmp_path / "native_ctx.json"
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "grep", "arguments": "{}"}}],
        },
        {"role": "tool", "content": "hits", "tool_name": "grep", "tool_call_id": "c1"},
        {"role": "assistant", "content": "done"},
    ]
    save_context_bundle(str(p), messages, "q", "done", True)
    loaded = load_context_messages(str(p))
    assert loaded == messages
    normalized = normalize_transcript_messages(loaded)
    assert normalized[1]["tool_calls"]
    assert normalized[2]["tool_name"] == "grep"
    assert normalized[2]["tool_call_id"] == "c1"


def test_load_context_file_reads_bundle(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(
        json.dumps(
            {
                "version": 1,
                "messages": [{"role": "user", "content": "prior"}],
            }
        ),
        encoding="utf-8",
    )
    assert load_context_messages(str(p)) == [{"role": "user", "content": "prior"}]
