"""Parse ``/mcp`` REPL commands (server list / add / remove / help)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from agentlib.mcp.cluster import sanitize_mcp_server_name

MCP_REPL_HELP = (
    "MCP — Model Context Protocol servers (prefs: agent.mcp_enabled, agent.mcp_servers).\n"
    "Changes apply in-memory until you run /set save.\n"
    "Process-wide: /mcp enable starts shared server connections (one cluster per AgentApp).\n"
    "Per session: /mcp session on|off adds or removes mcp_* ids in this session's enabled_tools "
    "(fork copies the parent's set). /set tools lists MCP tools with on/off per session.\n\n"
    "Commands:\n"
    "  /mcp help              Show this text\n"
    "  /mcp list              Show mcp_enabled + configured servers (JSON)\n"
    "  /mcp status            Prefs, errors, discovered tools, and this session's MCP enable count\n"
    "  /mcp enable | disable  Connect or disconnect shared MCP servers (agent.mcp_enabled)\n"
    "  /mcp session on | off  Enable or disable discovered MCP tools for this session only\n"
    "  /mcp reload            Schedule reconnect in the background (REPL returns immediately)\n"
    "  /mcp remove NAME       Drop a server entry by name\n"
    "  /mcp add stdio NAME [--cwd DIR] [--framing ndjson|content-length] COMMAND [ARGS...]\n"
    "                         Local subprocess (JSON-RPC over stdio).\n"
    "                         Framing: use --framing ndjson for Python MCP SDK / FastMCP servers (one JSON object per line);\n"
    "                         default content-length matches TypeScript / LSP-style clients (npx servers, etc.).\n"
    "                         For Python servers, COMMAND should be the venv interpreter that has `mcp` installed\n"
    "                         (e.g. ./.venv/bin/python3), not system `python`, or imports fail before MCP starts.\n"
    "                         Example:\n"
    "                           /mcp add stdio pyfs --framing ndjson --cwd /proj ./.venv/bin/python3 -m mymcp.server\n"
    "                           /mcp add stdio fs --cwd /tmp npx -y @modelcontextprotocol/server-filesystem /tmp\n"
    "  /mcp add http NAME URL [--header KEY=VALUE ...]\n"
    "                         Remote JSON-RPC POST to URL (simple request/response servers).\n"
    "                         Example:\n"
    "                           /mcp add http demo https://example.com/mcp \\\n"
    "                             --header \"Authorization=Bearer YOUR_TOKEN\"\n\n"
    "Tool ids look like: mcp_<server>_<tool_name> (see /set tools describe <id>).\n"
    "Also: /set tools reload refreshes MCP together with plugin toolsets.\n"
)


def parse_mcp_add_stdio_tokens(toks: List[str]) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    ``toks`` is shlex-split including leading ``/mcp``, ``add``, ``stdio``.

    Shape: ``/mcp add stdio NAME [--cwd DIR] [--framing ndjson|content-length] COMMAND [ARGS...]``
    """
    if len(toks) < 5:
        return (
            None,
            "Usage: /mcp add stdio NAME [--cwd DIR] [--framing ndjson|content-length] COMMAND [ARGS...]\n"
            "Example: /mcp add stdio fs npx -y @modelcontextprotocol/server-filesystem /tmp",
        )
    try:
        name = sanitize_mcp_server_name(toks[3])
    except ValueError as e:
        return None, str(e)
    i = 4
    cwd: Optional[str] = None
    framing = "content-length"
    while i < len(toks):
        if i + 1 < len(toks) and toks[i] == "--cwd":
            cwd = toks[i + 1]
            i += 2
            continue
        if i + 1 < len(toks) and toks[i] == "--framing":
            v = toks[i + 1].strip().lower().replace("_", "-")
            if v in ("ndjson", "jsonl", "newline"):
                framing = "ndjson"
            elif v in ("content-length", "lsp", "length"):
                framing = "content-length"
            else:
                return None, f"invalid --framing {toks[i + 1]!r} (use ndjson or content-length)"
            i += 2
            continue
        break
    argv = toks[i:]
    if not argv:
        return None, "missing COMMAND after NAME (and optional --cwd / --framing)"
    spec: Dict[str, Any] = {"name": name, "transport": "stdio", "command": argv[0], "args": argv[1:]}
    if cwd:
        spec["cwd"] = cwd
    if framing != "content-length":
        spec["stdio_framing"] = framing
    return spec, ""


def parse_mcp_add_http_tokens(toks: List[str]) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    ``/mcp add http NAME [--header KEY=VALUE ...] URL``

    URL must be the last token (quote it if it contains spaces).
    """
    if len(toks) < 6:
        return (
            None,
            "Usage: /mcp add http NAME [--header KEY=VALUE ...] URL\n"
            "Example: /mcp add http srv https://host.example/mcp",
        )
    try:
        name = sanitize_mcp_server_name(toks[3])
    except ValueError as e:
        return None, str(e)
    headers: Dict[str, str] = {}
    i = 4
    while i < len(toks):
        if toks[i] == "--header" and i + 1 < len(toks):
            pair = toks[i + 1]
            if "=" not in pair:
                return None, f"--header expects KEY=VALUE, got {pair!r}"
            k, _, v = pair.partition("=")
            k0, v0 = k.strip(), v.strip()
            if not k0:
                return None, f"invalid --header {pair!r}"
            headers[k0] = v0
            i += 2
            continue
        break
    rest = toks[i:]
    if len(rest) != 1:
        return (
            None,
            "expected a single URL after optional --header flags "
            "(quote the URL; no extra tokens after it).",
        )
    url = rest[0].strip()
    if not url.startswith(("http://", "https://")):
        return None, "URL must start with http:// or https://"
    spec: Dict[str, Any] = {"name": name, "transport": "http", "url": url}
    if headers:
        spec["headers"] = headers
    return spec, ""
