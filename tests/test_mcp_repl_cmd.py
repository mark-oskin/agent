from agentlib.mcp.repl_cmd import parse_mcp_add_http_tokens, parse_mcp_add_stdio_tokens


def test_parse_mcp_add_stdio_basic():
    toks = ["/mcp", "add", "stdio", "fs", "npx", "-y", "pkg", "/tmp"]
    spec, err = parse_mcp_add_stdio_tokens(toks)
    assert not err
    assert spec == {"name": "fs", "transport": "stdio", "command": "npx", "args": ["-y", "pkg", "/tmp"]}


def test_parse_mcp_add_stdio_with_cwd():
    toks = ["/mcp", "add", "stdio", "fs", "--cwd", "/tmp", "npx", "-y", "pkg", "/tmp"]
    spec, err = parse_mcp_add_stdio_tokens(toks)
    assert not err
    assert spec["cwd"] == "/tmp"
    assert spec["command"] == "npx"


def test_parse_mcp_add_stdio_with_framing_ndjson():
    toks = [
        "/mcp",
        "add",
        "stdio",
        "desk",
        "--cwd",
        "/proj",
        "--framing",
        "ndjson",
        "/proj/.venv/bin/macos-native-mcp",
    ]
    spec, err = parse_mcp_add_stdio_tokens(toks)
    assert not err
    assert spec["stdio_framing"] == "ndjson"
    assert spec["cwd"] == "/proj"
    assert spec["command"] == "/proj/.venv/bin/macos-native-mcp"


def test_parse_mcp_add_http_with_headers():
    toks = ["/mcp", "add", "http", "srv", "--header", "Authorization=Bearer x", "https://h.example/mcp"]
    spec, err = parse_mcp_add_http_tokens(toks)
    assert not err
    assert spec["url"] == "https://h.example/mcp"
    assert spec["headers"]["Authorization"] == "Bearer x"


def test_mcp_registry_records_errors_after_install():
    from agentlib.tools import mcp_registry

    class FakeCluster:
        tool_index = {}
        prompt_docs = {}
        connect_errors = ["e1"]

    mcp_registry.install(FakeCluster(), prefs_enabled=True)
    assert mcp_registry.last_connect_errors() == ["e1"]
