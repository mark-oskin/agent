"""--load_context / --save_context JSON helpers and argv wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import agent as d


def test_parse_context_messages_bare_list():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert d._parse_context_messages_data(msgs) == msgs


def test_parse_context_messages_wrapped():
    data = {"messages": [{"role": "user", "content": "a"}], "extra": 1}
    assert d._parse_context_messages_data(data) == [{"role": "user", "content": "a"}]


def test_parse_context_invalid_role():
    with pytest.raises(ValueError, match="invalid role"):
        d._parse_context_messages_data([{"role": "nope", "content": "x"}])


def test_save_and_reload_roundtrip(tmp_path: Path):
    p = tmp_path / "ctx.json"
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": '{"action":"answer","answer":"one"}'},
    ]
    d._save_context_bundle(str(p), messages, "follow-up", "two", True)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["user_query"] == "follow-up"
    assert data["final_answer"] == "two"
    assert data["answered"] is True
    assert len(data["messages"]) == 2
    loaded = d._load_context_messages(str(p))
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
    assert d._load_context_messages(str(p)) == [{"role": "user", "content": "prior"}]
