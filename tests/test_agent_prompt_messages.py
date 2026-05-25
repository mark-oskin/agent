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
    assert "Native tool use" in system
    assert "Answering:" in system
    assert "plain text is preferred" in system


def test_native_tool_prompt_uses_function_calling_not_json_tool_shape():
    system = prompts.build_agent_system_message(
        today_str="2026-05-18 (Monday)",
        second_opinion=False,
        cloud=False,
        primary_profile=None,
        reviewer_ollama_model=None,
        reviewer_hosted_profile=None,
        enabled_tools=frozenset({"search_web", "grep", "use_git"}),
        system_instruction_override=None,
        skill_suffix=None,
        ollama_model="test:latest",
        hosted_review_ready=lambda *_a, **_k: False,
        tool_policy_runner_text=lambda _et: "",
        tool_call_mode="native",
    )
    assert "Native function tools" in system
    assert "tool_calls on every tool call" in system
    assert "JSON-only tools" in system
    assert "search_web" in system
    assert "grep" in system
    assert "use_git" in system
    assert system.index("Native function tools") < system.index("JSON-only tools")
    assert "Final answers may be plain text" in system


def test_effective_system_prompt_upgrades_template_override_for_native():
    json_template = prompts.resolve_prompt_template_text(
        "coding",
        {
            "coding": {
                "kind": "overlay",
                "text": "Focus on minimal diffs.",
            }
        },
    )
    assert json_template
    body = prompts.effective_system_instruction_text_for_tools(
        json_template,
        frozenset({"search_web", "grep", "use_git"}),
        tool_call_mode="native",
        primary_profile=None,
    )
    assert "Native tool use" in body
    assert "Native function tools" in body
    assert "JSON-only tools" in body
    assert "Tool calls use this shape" not in body
    assert "Focus on minimal diffs." in body


def test_effective_system_prompt_show_matches_native_via_tool_call_mode():
    body = prompts.effective_system_instruction_text_for_tools(
        None,
        frozenset({"search_web", "read_file"}),
        tool_call_mode="native",
        primary_profile=None,
    )
    assert "Native function tools" in body
    assert "search_web" in body
    assert "read_file" in body
    assert '{"action":"tool_call","tool":"search_web"' not in body.split("JSON-only tools")[0]


def test_native_tool_prompt_disabled_for_hosted_primary():
    from agentlib.llm.profile import LlmProfile

    system = prompts.build_agent_system_message(
        today_str="2026-05-18 (Monday)",
        second_opinion=False,
        cloud=True,
        primary_profile=LlmProfile(backend="hosted", base_url="https://x/v1", model="m", api_key="k"),
        reviewer_ollama_model=None,
        reviewer_hosted_profile=None,
        enabled_tools=frozenset({"search_web"}),
        system_instruction_override=None,
        skill_suffix=None,
        ollama_model="test:latest",
        hosted_review_ready=lambda *_a, **_k: False,
        tool_policy_runner_text=lambda _et: "",
        tool_call_mode="native",
    )
    assert "Native function tools" not in system
    assert '{"action":"tool_call","tool":"search_web"' in system


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
