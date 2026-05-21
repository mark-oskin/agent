"""Generation rate tracker and streaming gen_rate events."""

from __future__ import annotations

import json
import time

from agentlib.llm import streaming as llm_streaming
from agentlib.llm.gen_rate import GenRateTracker, estimate_tokens_from_text


def test_estimate_tokens_from_text():
    assert estimate_tokens_from_text("") == 0
    assert estimate_tokens_from_text("abcd") == 1
    assert estimate_tokens_from_text("a" * 8) == 2


def test_gen_rate_tracker_interval_sampling():
    tr = GenRateTracker()
    tr.add_tokens(20)
    time.sleep(1.0)
    rate = tr.sample_interval(min_elapsed=0.5)
    assert rate is not None
    assert 12 <= rate <= 28


def test_gen_rate_tracker_ignores_tiny_interval():
    tr = GenRateTracker()
    tr.add_tokens(10)
    assert tr.sample_interval(min_elapsed=1.0) is None
    time.sleep(1.1)
    rate = tr.sample_interval(min_elapsed=0.5)
    assert rate is not None
    assert 5 <= rate <= 15


def test_merge_stream_emits_gen_rate_with_ollama_eval_count(monkeypatch):
    emitted: list[dict] = []

    def cap(ev: dict) -> None:
        emitted.append(dict(ev))

    monkeypatch.setattr(llm_streaming, "sink_emit", cap)
    llm_streaming.reset_assistant_answer_streamed()

    lines = [
        json.dumps(
            {
                "message": {"content": '{"action":"answer","answer":"hi'},
                "done": False,
                "eval_count": 1,
            }
        ),
        json.dumps(
            {
                "message": {"content": '"}'},
                "done": True,
                "eval_count": 3,
                "eval_duration": 500_000_000,
            }
        ),
    ]
    llm_streaming.merge_stream_message_chunks(
        iter(lines),
        stream_user_visible=True,
        agent_stream_thinking_enabled=lambda: False,
    )
    rates = [e["tok_per_sec"] for e in emitted if e.get("type") == "gen_rate"]
    assert all(isinstance(r, (int, float)) and r >= 0 for r in rates)


def test_record_thinking_chunk_tokens_for_gen_rate():
    vis = llm_streaming._VisibleStreamState()
    llm_streaming._record_thinking_chunk_tokens(
        "reasoning tokens here",
        vis,
        stream_user_visible=True,
    )
    time.sleep(0.3)
    rate = vis.gen_rate.sample_interval(min_elapsed=0.2)
    assert rate is not None
    assert rate > 0


def test_record_thinking_skipped_when_ollama_eval_count_active():
    vis = llm_streaming._VisibleStreamState()
    vis.has_ollama_eval = True
    llm_streaming._record_thinking_chunk_tokens(
        "should not count",
        vis,
        stream_user_visible=True,
    )
    assert vis.gen_rate._tokens_in_period == 0
