"""Chars-per-token estimator calibration and usage notes."""

from __future__ import annotations

import json

from agentlib.llm import streaming as llm_streaming
from agentlib.llm.token_estimate import (
    DEFAULT_CHARS_PER_TOKEN,
    CharsPerTokenEstimator,
    estimate_tokens_from_text,
)
from agentlib.llm import usage as llm_usage


def test_default_estimate_uses_default_chars_per_token():
    est = CharsPerTokenEstimator()
    assert est.chars_per_token == DEFAULT_CHARS_PER_TOKEN
    assert estimate_tokens_from_text("", estimator=est) == 0
    assert estimate_tokens_from_text("abcd", estimator=est) == 1
    assert estimate_tokens_from_text("a" * 12, estimator=est) == 3


def test_observe_ollama_completion_ema():
    est = CharsPerTokenEstimator()
    # 45 chars / 10 tokens => 4.5 observed
    est.observe_ollama_completion(content="x" * 45, thinking="", eval_count=10)
    assert est.calibration_observations == 1
    assert est.chars_per_token == 4.5
    # 120 chars / 10 tokens => 12.0 observed, clamped to 12, blended with EMA
    est.observe_ollama_completion(content="z" * 120, thinking="", eval_count=10)
    assert est.calibration_observations == 2
    assert est.last_observed_chars_per_token == 12.0
    assert est.chars_per_token > 4.5
    assert est.chars_per_token <= 12.0


def test_observe_skips_small_completions():
    est = CharsPerTokenEstimator()
    est.observe_ollama_completion(content="hi", thinking="", eval_count=100)
    assert est.calibration_observations == 0
    est.observe_ollama_completion(content="x" * 30, thinking="", eval_count=3)
    assert est.calibration_observations == 0


def test_usage_includes_calibration_note():
    est = CharsPerTokenEstimator()
    text = llm_usage.format_last_ollama_usage_for_repl(None, est)
    assert "chars/token" in text
    assert "not calibrated yet" in text
    est.observe_ollama_completion(content="a" * 48, thinking="", eval_count=8)
    text2 = llm_usage.format_last_ollama_usage_for_repl({"eval_count": 8}, est)
    assert "calibrated from 1" in text2


def test_stream_calibration_updates_estimator():
    est = CharsPerTokenEstimator()
    content = '{"action":"answer","answer":"' + ("hello world " * 8) + '"}'
    lines = [
        json.dumps({"message": {"content": content}, "done": False, "eval_count": 4}),
        json.dumps(
            {
                "message": {"content": ""},
                "done": True,
                "eval_count": 20,
                "eval_duration": 1_000_000_000,
            }
        ),
    ]
    llm_streaming.merge_stream_message_chunks(
        iter(lines),
        stream_user_visible=False,
        agent_stream_thinking_enabled=lambda: False,
        chars_per_token_estimator=est,
    )
    assert est.calibration_observations >= 1
