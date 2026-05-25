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
    assert names == ["grep", "read_file", "search_web", "use_git"]


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


def test_use_git_call_python_native_schemas():
    for tid in ("use_git", "call_python", "run_applescript", "download_file", "tail_file"):
        tool = ollama_function_tool_definition(tid)
        assert tool is not None
        assert tool["function"]["name"] == tid


def test_second_opinion_native_schema():
    tool = ollama_function_tool_definition("second_opinion")
    assert tool is not None
    assert "draft_answer" in tool["function"]["parameters"]["properties"]


def test_ollama_tools_includes_plugins_when_enabled():
    from agentlib.tools.plugins import PLUGIN_TOOL_HANDLERS

    if not PLUGIN_TOOL_HANDLERS:
        return
    tid = next(iter(PLUGIN_TOOL_HANDLERS))
    tools = ollama_tools_for_enabled(frozenset({tid}))
    assert any(t["function"]["name"] == tid for t in tools)


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


def test_invalid_agent_response_user_content_native_vs_json():
    from agentlib.llm.tool_schemas import invalid_agent_response_user_content

    native = invalid_agent_response_user_content(tool_call_mode="native")
    assert "plain text" in native
    assert "function-calling API" in native
    assert "not valid agent JSON" not in native

    json_msg = invalid_agent_response_user_content(tool_call_mode="json")
    assert "not valid agent JSON" in json_msg
    assert "Respond with JSON only" in json_msg


def test_infer_bare_args_maps_common_native_shapes():
    from agentlib.agent_json import infer_tool_call_from_bare_args, parse_agent_json
    from agentlib.agent_json import AgentJsonDeps
    from agentlib.tools import turn_support
    from agentlib.tools.registry import ToolRegistry
    import os

    project = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    reg = ToolRegistry(default_tools_dir=os.path.join(project, "tools"))
    reg.load_plugin_toolsets(reg.default_tools_dir)
    reg.register_aliases()
    known = reg.all_known_tools()
    deps = AgentJsonDeps(
        all_known_tools=reg.all_known_tools,
        coerce_enabled_tools=reg.coerce_enabled_tools,
        merge_tool_param_aliases=turn_support.merge_tool_param_aliases,
    )

    q = infer_tool_call_from_bare_args({"query": "Seattle Mariners schedule"}, known)
    assert q["tool"] == "search_web"
    assert q["parameters"]["query"] == "Seattle Mariners schedule"

    u = infer_tool_call_from_bare_args({"url": "https://example.com"}, known)
    assert u["tool"] == "fetch_page"

    parsed = parse_agent_json('{"query": "who is president"}', deps)
    assert parsed["action"] == "tool_call"
    assert parsed["tool"] == "search_web"


def test_tool_result_user_message_native_allows_plain_text():
    from agentlib.tools import turn_support

    msg = turn_support.tool_result_user_message(
        "search_web",
        {"query": "q"},
        "results",
        tool_output_max=1000,
        native_transport=True,
    )
    assert "plain text" in msg
    assert "native tool_calls" in msg
    assert "respond with JSON only" not in msg
