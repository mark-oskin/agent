"""Second opinion (Ollama + cloud) and next_action / rationale wiring."""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout

from tests.harness import build_test_app
from agentlib.settings import AgentSettings


def test_second_opinion_disabled_nudges_finalize(monkeypatch):
    fin = json.dumps(
        {
            "action": "answer",
            "answer": "done",
            "next_action": "finalize",
            "rationale": "flags off",
        }
    )
    out = _run_with_seq(
        monkeypatch,
        ["hi"],
        [
            json.dumps(
                {
                    "action": "answer",
                    "answer": "draft",
                    "next_action": "second_opinion",
                    "rationale": "unsure",
                }
            ),
            fin,
            fin,
        ],
        second_opinion=False,
        cloud=False,
        stub_plain=None,
    )
    assert out == "done"


def test_second_opinion_ollama_invokes_reviewer(monkeypatch):
    plain_models = []

    def fake_plain(messages, model):
        plain_models.append(model)
        return "Second opinion: looks reasonable."

    out = _run_with_seq(
        monkeypatch,
        ["--second-opinion", "Hello"],
        [
            json.dumps(
                {
                    "action": "answer",
                    "answer": "first draft",
                    "next_action": "second_opinion",
                    "rationale": "want review",
                    "second_opinion_backend": "ollama",
                }
            ),
            json.dumps(
                {
                    "action": "answer",
                    "answer": "final",
                    "next_action": "finalize",
                    "rationale": "merged",
                }
            ),
        ],
        second_opinion=True,
        cloud=False,
        stub_plain=fake_plain,
    )
    assert out == "final"
    assert plain_models and plain_models[0]


def test_second_opinion_requires_rationale(monkeypatch):
    fin = json.dumps(
        {
            "action": "answer",
            "answer": "fixed",
            "next_action": "finalize",
            "rationale": "ok",
        }
    )
    out = _run_with_seq(
        monkeypatch,
        ["--second-opinion", "q"],
        [
            json.dumps({"action": "answer", "answer": "x", "next_action": "second_opinion"}),
            fin,
            fin,
        ],
        second_opinion=True,
        cloud=False,
        stub_plain=lambda m, mod: "ok",
    )
    assert out == "fixed"


def test_second_opinion_cloud_invokes_openai(monkeypatch):
    called = []

    def fake_openai(messages):
        called.append(messages)
        return "Cloud reviewer: add caveat on edge case."

    fin = json.dumps(
        {
            "action": "answer",
            "answer": "final",
            "next_action": "finalize",
            "rationale": "done",
        }
    )
    out = _run_with_seq(
        monkeypatch,
        ["--cloud-ai", "Q"],
        [
            json.dumps(
                {
                    "action": "answer",
                    "answer": "draft",
                    "next_action": "second_opinion",
                    "rationale": "check",
                    "second_opinion_backend": "openai",
                }
            ),
            fin,
            fin,
        ],
        second_opinion=False,
        cloud=True,
        stub_plain=None,
        stub_openai=fake_openai,
    )
    assert out == "final"
    assert len(called) == 1


def _run_with_seq(
    monkeypatch,
    argv_tail,
    responses,
    *,
    second_opinion: bool,
    cloud: bool,
    stub_plain,
    stub_openai=None,
):
    call_i = {"i": 0}

    def fake_chat(
        messages, primary_profile=None, enabled_tools=None, verbose=0, **kwargs
    ):  # noqa: ARG001
        idx = call_i["i"]
        call_i["i"] += 1
        if idx >= len(responses):
            return json.dumps({"action": "error", "error": f"no response #{idx}"})
        return responses[idx]

    app = build_test_app(monkeypatch)
    app.settings = AgentSettings.defaults()
    monkeypatch.setattr(app, "call_ollama_chat", fake_chat)
    # Provide a placeholder API key when cloud-ai is enabled so hosted reviewer paths are "ready".
    if cloud:
        app.settings.set(("openai", "api_key"), "sk-test-placeholder")
    monkeypatch.setattr(app, "route_requires_websearch", lambda *a, **k: None)
    monkeypatch.setattr(app, "route_requires_websearch_after_answer", lambda *a, **k: None)
    if stub_plain is not None:
        monkeypatch.setattr(app, "call_ollama_plaintext", stub_plain)
    if stub_openai is not None:
        monkeypatch.setattr(app, "call_openai_chat_plain", stub_openai)

    pre: list[str] = []
    if second_opinion:
        pre.append("--second-opinion")
    if cloud:
        pre.append("--cloud-ai")
    buf = io.StringIO()
    with redirect_stdout(buf):
        app.run(pre + list(argv_tail))
    return buf.getvalue().strip()
