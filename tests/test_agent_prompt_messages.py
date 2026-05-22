"""Agent system vs user message split for LLM API calls."""

from __future__ import annotations

from agentlib import prompts


def test_interactive_turn_user_content_is_slim():
    user = prompts.interactive_turn_user_content("What is 2+2?")
    assert user == "User request:\nWhat is 2+2?"
    assert "universal agent" not in user.lower()


def test_build_agent_system_message_includes_contract():
    system = prompts.build_agent_system_message(
        today_str="2026-05-18 (Monday)",
        second_opinion=False,
        cloud=False,
        primary_profile=None,
        reviewer_ollama_model=None,
        reviewer_hosted_profile=None,
        enabled_tools=frozenset(),
        system_instruction_override=None,
        skill_suffix=None,
        ollama_model="test:latest",
        hosted_review_ready=lambda *_a, **_k: False,
        tool_policy_runner_text=lambda _et: "",
    )
    assert "You are a universal agent" in system
    assert "2026-05-18" in system
    assert "Respond with JSON only" in system


def test_messages_for_agent_api_call_prepends_single_system():
    legacy_user = prompts.interactive_turn_user_message(
        user_query="hello",
        today_str="2026-05-18",
        second_opinion=False,
        cloud=False,
        primary_profile=None,
        reviewer_ollama_model=None,
        reviewer_hosted_profile=None,
        enabled_tools=frozenset(),
        system_instruction_override=None,
        skill_suffix=None,
        ollama_model="m",
        hosted_review_ready=lambda *_a, **_k: False,
        tool_policy_runner_text=lambda _et: "",
    )
    transcript = [{"role": "user", "content": legacy_user}]
    system = "SYSTEM_BLOCK"
    api = prompts.messages_for_agent_api_call(transcript, system)
    assert api[0] == {"role": "system", "content": "SYSTEM_BLOCK"}
    assert api[1]["role"] == "user"
    assert api[1]["content"] == "User request:\nhello"
    assert sum(1 for m in api if m["role"] == "system") == 1


def test_strip_legacy_agent_turn_user_content():
    legacy = (
        "You are a universal agent.\n\n"
        "Today's date (system clock): 2026-01-01\n\n"
        "User request:\nlegacy question\n\n"
        "Respond with JSON only. No other text."
    )
    assert prompts.strip_legacy_agent_turn_user_content(legacy) == "User request:\nlegacy question"
