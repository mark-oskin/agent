"""``request_options`` on ``LlmProfile``, merge helpers, and REPL persistence surface."""

from __future__ import annotations

import pytest

from agentlib.llm.profile import (
    LlmProfile,
    llm_profile_from_pref,
    llm_profile_to_pref,
    preserved_request_options,
)
from agentlib.llm.request_options import (
    merge_hosted_request_options,
    merge_ollama_options_payload,
    normalize_request_options_pref,
)

from tests.harness import build_test_session


def test_llm_profile_pref_round_trip_hosted_and_ollama():
    h = LlmProfile(
        backend="hosted",
        base_url="https://api.example.com/v1",
        model="gpt-test",
        api_key="k",
        request_options={"temperature": 0.2, "top_p": 0.95, "num_ctx": 32768},  # num_ctx only Ollama-merged
    )
    d = llm_profile_to_pref(h)
    assert d["request_options"]["temperature"] == 0.2
    assert d["request_options"]["top_p"] == 0.95
    assert d["request_options"]["num_ctx"] == 32768
    h2 = llm_profile_from_pref(d)
    assert h2 is not None and h2.backend == "hosted"
    assert h2.request_options["temperature"] == 0.2

    o = LlmProfile(backend="ollama", request_options={"num_ctx": 4096})
    d2 = llm_profile_to_pref(o)
    assert d2["backend"] == "ollama"
    assert d2["request_options"]["num_ctx"] == 4096
    o2 = llm_profile_from_pref(d2)
    assert o2 is not None and o2.backend == "ollama"
    assert o2.request_options["num_ctx"] == 4096


def test_llm_profile_to_pref_omits_empty_request_options():
    p = LlmProfile(backend="ollama")
    assert "request_options" not in llm_profile_to_pref(p)


def test_preserved_request_options_copy():
    p = LlmProfile(backend="hosted", base_url="https://x/v1", model="m", request_options={"seed": 1})
    c = preserved_request_options(p)
    assert c == {"seed": 1}
    p.request_options.clear()
    assert c["seed"] == 1


def test_merge_hosted_skips_reserved_and_unknown():
    body = {"model": "m", "messages": []}
    merge_hosted_request_options(
        body,
        {"model": "evil", "messages": [], "temperature": 0.5, "bogus": 1, "max_tokens": 10},
        default_temperature=None,
    )
    assert body["model"] == "m"
    assert body["temperature"] == 0.5
    assert body["max_tokens"] == 10
    assert "bogus" not in body


def test_merge_ollama_options_payload_nested():
    payload: dict = {"model": "q"}
    merge_ollama_options_payload(payload, {"temperature": 0.4, "num_ctx": 8192})
    assert payload["options"]["temperature"] == 0.4
    assert payload["options"]["num_ctx"] == 8192


def test_normalize_pref_drops_illegal_nested():
    blob = normalize_request_options_pref({"temperature": 0.1, "weird": {"a": 1}})
    assert blob == {"temperature": 0.1}


@pytest.mark.parametrize("cmd", ["/set", "/settings"])
def test_session_primary_request_options_and_backend_swap_preserves(monkeypatch, cmd: str):
    _, session = build_test_session(monkeypatch)
    session.execute_line(f"{cmd} primary request_options set temperature 0.55")
    assert session.primary_profile.request_options.get("temperature") == pytest.approx(0.55)
    session.execute_line(
        f"{cmd} primary llm hosted https://api.example.com/v1 hosted-model sekrit"
    )
    assert session.primary_profile.backend == "hosted"
    assert session.primary_profile.request_options.get("temperature") == pytest.approx(0.55)
    session.execute_line(f"{cmd} primary llm ollama")
    assert session.primary_profile.backend == "ollama"
    assert session.primary_profile.request_options.get("temperature") == pytest.approx(0.55)
