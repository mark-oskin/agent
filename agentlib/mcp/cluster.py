"""Connect to configured MCP servers and expose tool discovery + invocation."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

from agentlib.mcp.format import format_tool_result
from agentlib.mcp.http_session import HttpMcpSession
from agentlib.mcp.stdio_session import StdioMcpSession

from agentlib.settings import AgentSettings


def _sanitize_server_name(raw: str) -> str:
    s = (raw or "").strip().lower().replace("-", "_")
    if not re.match(r"^[a-z][a-z0-9_]{0,63}$", s):
        raise ValueError(f"invalid MCP server name {raw!r} (use a-z, digits, underscore; start with a letter)")
    return s


def sanitize_mcp_server_name(raw: str) -> str:
    """Normalize and validate the ``name`` field for MCP server entries (prefs / ``/mcp``)."""
    return _sanitize_server_name(raw)


def merge_stdio_mcp_child_env(raw_env: object) -> Dict[str, str]:
    """Environment for stdio MCP subprocesses.

    Python runtimes often use fully buffered stdout when connected to a pipe; MCP
    responses then never arrive until the buffer fills, causing client timeouts on
    ``initialize``. Set ``PYTHONUNBUFFERED=1`` unless the server entry's ``env``
    explicitly defines ``PYTHONUNBUFFERED`` (so hosts can opt out).
    """
    env_merged: Dict[str, str] = {**os.environ}
    explicit_unbuf = False
    if isinstance(raw_env, dict):
        for k, v in raw_env.items():
            ks = str(k)
            env_merged[ks] = str(v)
            if ks.strip().upper().replace("-", "_") == "PYTHONUNBUFFERED":
                explicit_unbuf = True
    if not explicit_unbuf:
        env_merged["PYTHONUNBUFFERED"] = "1"
    return env_merged


def inject_python_u_flag(argv: List[str]) -> List[str]:
    """Insert ``-u`` after a CPython launcher when missing (stdio MCP reliability)."""
    if len(argv) < 1 or "-u" in argv:
        return argv
    exe = os.path.basename(argv[0]).strip().lower()
    if not exe:
        return argv
    if exe in ("python", "python2", "python3") or exe.startswith("python3.") or exe.startswith("python2."):
        return [argv[0], "-u"] + list(argv[1:])
    return argv


def _sanitize_tool_suffix(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", (name or "").strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower() or "tool"


def composite_tool_id(server: str, mcp_tool_name: str, used: set[str]) -> str:
    base = f"mcp_{server}_{_sanitize_tool_suffix(mcp_tool_name)}"
    if base not in used:
        used.add(base)
        return base
    n = 2
    while f"{base}_{n}" in used:
        n += 1
    cid = f"{base}_{n}"
    used.add(cid)
    return cid


def _prompt_line_for_tool(composite_id: str, server: str, rec: Dict[str, Any]) -> str:
    desc = str(rec.get("description") or "").strip()
    schema = rec.get("inputSchema")
    schema_note = ""
    if isinstance(schema, dict):
        try:
            schema_note = json.dumps(schema, ensure_ascii=False)
        except Exception:
            schema_note = str(schema)
        if len(schema_note) > 420:
            schema_note = schema_note[:400] + "…"
    bits = [
        f"{composite_id} — MCP tool from server {server!r} (original name {rec.get('name')!r}).",
    ]
    if desc:
        bits.append(desc)
    if schema_note:
        bits.append(f"parameters JSON must satisfy inputSchema: {schema_note}")
    return " ".join(bits)


class _Conn:
    __slots__ = ("session",)

    def __init__(self, session: Any):
        self.session = session


class McpCluster:
    """Live MCP connections for one sync cycle."""

    def __init__(self) -> None:
        self._by_server: Dict[str, _Conn] = {}
        self.tool_index: Dict[str, Tuple[str, str]] = {}
        self.prompt_docs: Dict[str, str] = {}
        self.input_schemas: Dict[str, dict] = {}
        self.connect_errors: List[str] = []

    def disconnect_all(self) -> None:
        for c in list(self._by_server.values()):
            try:
                c.session.close()
            except Exception:
                pass
        self._by_server.clear()

    @staticmethod
    def build_from_settings(settings: AgentSettings) -> "McpCluster":
        cluster = McpCluster()
        servers = settings.get(("agent", "mcp_servers"))
        if not isinstance(servers, list) or not servers:
            return cluster
        used_ids: set[str] = set()
        for i, raw in enumerate(servers):
            if not isinstance(raw, dict):
                cluster.connect_errors.append(f"mcp_servers[{i}]: entry must be an object")
                continue
            try:
                name = _sanitize_server_name(str(raw.get("name") or ""))
            except ValueError as e:
                cluster.connect_errors.append(f"mcp_servers[{i}]: {e}")
                continue
            transport = str(raw.get("transport") or "stdio").strip().lower()
            try:
                if transport in ("stdio", "local"):
                    cmd = raw.get("command")
                    args = raw.get("args") or []
                    if isinstance(cmd, list):
                        argv = [str(x) for x in cmd]
                    elif isinstance(cmd, str) and cmd.strip():
                        argv = [cmd.strip()]
                    else:
                        raise ValueError("stdio MCP server requires command (string or argv list)")
                    extra = [str(x) for x in args] if isinstance(args, list) else []
                    argv = inject_python_u_flag(argv + extra)
                    env_merged = merge_stdio_mcp_child_env(raw.get("env"))
                    cwd = raw.get("cwd")
                    cwd_s = str(cwd).strip() if cwd else None
                    raw_framing = raw.get("stdio_framing") if raw.get("stdio_framing") is not None else raw.get("framing")
                    framing_s = (
                        str(raw_framing).strip().lower().replace("_", "-") if raw_framing is not None else "content-length"
                    )
                    if framing_s in ("", "default"):
                        framing_s = "content-length"
                    if framing_s in ("jsonl", "newline"):
                        framing_s = "ndjson"
                    if framing_s not in ("content-length", "ndjson"):
                        raise ValueError(f"stdio_framing must be 'content-length' or 'ndjson', got {raw_framing!r}")
                    session = StdioMcpSession(
                        command=argv, env=env_merged, cwd=cwd_s or None, stdio_framing=framing_s
                    )
                elif transport in ("http", "https", "remote"):
                    url = str(raw.get("url") or "").strip()
                    if transport in ("http", "https") and url and "://" not in url:
                        url = f"{transport}://{url}"
                    if not url.startswith(("http://", "https://")):
                        raise ValueError("http MCP server requires url")
                    hdr = raw.get("headers")
                    headers = {str(k): str(v) for k, v in hdr.items()} if isinstance(hdr, dict) else {}
                    session = HttpMcpSession(url, headers=headers)
                else:
                    raise ValueError(f"unknown transport {transport!r} (use stdio or http)")
            except Exception as e:
                cluster.connect_errors.append(f"server {name!r}: {e}")
                continue
            if name in cluster._by_server:
                cluster.connect_errors.append(f"duplicate MCP server name {name!r}")
                try:
                    session.close()
                except Exception:
                    pass
                continue
            try:
                session.handshake()
                tools = session.list_tools()
            except Exception as e:
                tail = ""
                if hasattr(session, "drain_stderr_messages"):
                    lines = session.drain_stderr_messages()
                    if lines:
                        tail = " | stderr (last lines): " + " | ".join(lines[-6:])
                hint_nd = ""
                blob = (tail + str(e)).lower()
                if "content-length" in blob and "json_invalid" in blob:
                    hint_nd = (
                        " — for Python MCP SDK / FastMCP stdio, re-add with "
                        "`/mcp add stdio ... --framing ndjson ...` (NDJSON lines, not Content-Length)."
                    )
                cluster.connect_errors.append(f"server {name!r}: handshake/tools.list failed: {e}{tail}{hint_nd}")
                try:
                    session.close()
                except Exception:
                    pass
                continue
            cluster._by_server[name] = _Conn(session)
            for td in tools:
                if not isinstance(td, dict):
                    continue
                orig = str(td.get("name") or "").strip()
                if not orig:
                    continue
                cid = composite_tool_id(name, orig, used_ids)
                cluster.tool_index[cid] = (name, orig)
                cluster.prompt_docs[cid] = _prompt_line_for_tool(cid, name, td)
                schema = td.get("inputSchema")
                if isinstance(schema, dict):
                    cluster.input_schemas[cid] = schema
                else:
                    cluster.input_schemas[cid] = {"type": "object", "additionalProperties": True}
        return cluster

    def invoke_composite(self, composite_id: str, params: Dict[str, Any]) -> str:
        pair = self.tool_index.get(composite_id)
        if not pair:
            return f"MCP error: unknown MCP tool id {composite_id!r}"
        server, tool_name = pair
        conn = self._by_server.get(server)
        if not conn:
            return f"MCP error: server {server!r} is not connected"
        try:
            raw = conn.session.call_tool(tool_name, params if isinstance(params, dict) else {})
            return format_tool_result(raw)
        except Exception as e:
            extra = conn.session.drain_stderr_messages()
            tail = f"\nstderr (last lines): {extra}" if extra else ""
            return f"MCP error ({server}/{tool_name}): {e}{tail}"
