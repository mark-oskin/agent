import io

import pytest

from agentlib.mcp.cluster import McpCluster, composite_tool_id, inject_python_u_flag, merge_stdio_mcp_child_env
from agentlib.mcp.jsonrpc_compat import jsonrpc_response_id_matches
from agentlib.mcp.jsonrpc_framing import read_framed_message, write_framed_message
from agentlib.mcp.format import format_tool_result
from agentlib.settings import AgentSettings


def test_composite_tool_id_collision():
    used: set[str] = set()
    assert composite_tool_id("srv", "read-file", used) == "mcp_srv_read_file"
    assert composite_tool_id("srv", "read.file", used) == "mcp_srv_read_file_2"


def test_framed_jsonrpc_roundtrip():
    buf = io.BytesIO()
    write_framed_message(buf, {"jsonrpc": "2.0", "id": 1, "result": {"tools": []}})
    buf.seek(0)
    msg = read_framed_message(buf)
    assert msg["id"] == 1
    assert msg["result"]["tools"] == []


def test_ndjson_jsonrpc_roundtrip():
    from agentlib.mcp.jsonrpc_framing import read_ndjson_message, write_ndjson_message

    buf = io.BytesIO()
    write_ndjson_message(buf, {"jsonrpc": "2.0", "id": 2, "result": {"tools": [{"name": "x"}]}})
    buf.seek(0)
    msg = read_ndjson_message(buf)
    assert msg["id"] == 2
    assert msg["result"]["tools"][0]["name"] == "x"


def test_merge_stdio_mcp_child_env_sets_python_unbuffered():
    e = merge_stdio_mcp_child_env({})
    assert e.get("PYTHONUNBUFFERED") == "1"


def test_merge_stdio_mcp_child_env_user_override():
    assert merge_stdio_mcp_child_env({"PYTHONUNBUFFERED": "0"}).get("PYTHONUNBUFFERED") == "0"


def test_jsonrpc_response_id_matches_accepts_string_ids():
    assert jsonrpc_response_id_matches(1, 1)
    assert jsonrpc_response_id_matches("1", 1)
    assert not jsonrpc_response_id_matches("2", 1)
    assert not jsonrpc_response_id_matches(None, 1)


def test_inject_python_u_flag_inserts_after_python():
    assert inject_python_u_flag(["python3", "-m", "pkg"]) == ["python3", "-u", "-m", "pkg"]
    assert inject_python_u_flag(["./.venv/bin/python3", "-m", "x"]) == ["./.venv/bin/python3", "-u", "-m", "x"]
    assert inject_python_u_flag(["python3", "-u", "-m", "x"]) == ["python3", "-u", "-m", "x"]
    assert inject_python_u_flag(["node", "script.js"]) == ["node", "script.js"]


def test_stdio_mcp_session_fails_fast_when_child_exits_before_reply():
    from agentlib.mcp.stdio_session import StdioMcpSession

    s = StdioMcpSession(
        command=["/bin/sh", "-c", 'echo "ModuleNotFoundError: No module named '"'"'mcp'"'"'" >&2; exit 1']
    )
    try:
        with pytest.raises(RuntimeError, match="exited with code"):
            s.request("initialize", {"protocolVersion": "test"}, timeout_s=10.0)
    finally:
        s.close()


def test_format_tool_result_text_blocks():
    out = format_tool_result({"content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]})
    assert "hello" in out and "world" in out


def test_mcp_registry_install_disabled():
    from agentlib.tools import mcp_registry

    mcp_registry.install(None, prefs_enabled=False)
    assert not mcp_registry.union_into_session_enabled()
    assert len(mcp_registry.all_ids()) == 0


def test_settings_group_set_mcp_servers_json():
    s = AgentSettings.defaults()
    s.group_set("agent", "mcp_servers", '[{"name":"demo","transport":"http","url":"http://127.0.0.1:9/x"}]')
    raw = s.get(("agent", "mcp_servers"))
    assert isinstance(raw, list) and raw[0]["name"] == "demo"


def test_mcp_cluster_skips_invalid_server_entries():
    """build_from_settings should record errors, not raise, for bad rows."""
    s = AgentSettings.defaults()
    s.set(("agent", "mcp_enabled"), True)
    s.set(
        ("agent", "mcp_servers"),
        [
            {"name": "bad name!", "transport": "stdio", "command": "echo"},
            {"name": "ok", "transport": "http", "url": "http://127.0.0.1:1/nope"},
        ],
    )
    c = McpCluster.build_from_settings(s)
    assert c.connect_errors
    c.disconnect_all()
