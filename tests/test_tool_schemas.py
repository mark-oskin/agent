"""Native tool schema registry (Phase 1)."""

from agentlib.llm.tool_schemas import (
    NATIVE_PHASE1_TOOL_IDS,
    ollama_function_tool_definition,
    ollama_tools_for_enabled,
)


def test_phase1_search_web_schema():
    tool = ollama_function_tool_definition("search_web")
    assert tool is not None
    assert tool["type"] == "function"
    fn = tool["function"]
    assert fn["name"] == "search_web"
    assert "query" in fn["parameters"]["properties"]
    assert fn["parameters"]["required"] == ["query"]


def test_ollama_tools_for_enabled_filters_and_sorts():
    tools = ollama_tools_for_enabled(frozenset({"search_web", "grep", "read_file", "use_git"}))
    names = [t["function"]["name"] for t in tools]
    assert names == ["grep", "read_file", "search_web"]
    assert "use_git" not in names


def test_grep_list_directory_replace_text_native_schemas():
    for tid in ("grep", "list_directory", "replace_text", "search_web_fetch_top"):
        tool = ollama_function_tool_definition(tid)
        assert tool is not None
        assert tool["function"]["name"] == tid
    grep_fn = ollama_function_tool_definition("grep")["function"]
    assert grep_fn["parameters"]["required"] == ["pattern"]
    assert "path" in grep_fn["parameters"]["properties"]
    list_fn = ollama_function_tool_definition("list_directory")["function"]
    assert list_fn["parameters"]["required"] == ["path"]
    replace_fn = ollama_function_tool_definition("replace_text")["function"]
    assert replace_fn["parameters"]["required"] == ["path", "pattern", "replacement"]
    swft_fn = ollama_function_tool_definition("search_web_fetch_top")["function"]
    assert swft_fn["parameters"]["required"] == ["query"]
    assert "fetch_top_n" in swft_fn["parameters"]["properties"]


def test_ollama_tools_empty_when_none_enabled():
    assert ollama_tools_for_enabled(frozenset()) == []
    assert ollama_tools_for_enabled(None) == []


def test_phase1_tool_ids_cover_schemas():
    for tid in NATIVE_PHASE1_TOOL_IDS:
        assert ollama_function_tool_definition(tid) is not None


def test_web_search_required_user_content_native_vs_json():
    from agentlib.llm.tool_schemas import tool_call_only_nudge, web_search_required_user_content

    native = web_search_required_user_content("search_web", "who is president", tool_call_mode="native")
    assert "native tool 'search_web'" in native
    assert "JSON" in native
    assert "who is president" in native

    json_msg = web_search_required_user_content("search_web", "who is president", tool_call_mode="json")
    assert "Respond with JSON only in tool_call form" in json_msg

    assert "function-calling API" in tool_call_only_nudge(tool_call_mode="native")
    assert "JSON tool_call only" in tool_call_only_nudge(tool_call_mode="json")

    from agentlib.llm.profile import LlmProfile
    from agentlib.llm.tool_schemas import tool_transport_uses_native

    assert tool_transport_uses_native(
        tool_call_mode="native",
        primary_profile=LlmProfile(backend="hosted", base_url="https://x/v1", model="m", api_key="k"),
    ) is False
    hosted_msg = web_search_required_user_content(
        "search_web", "q", tool_call_mode="native", primary_profile=LlmProfile(backend="hosted", base_url="https://x/v1", model="m", api_key="k")
    )
    assert "Respond with JSON only in tool_call form" in hosted_msg
