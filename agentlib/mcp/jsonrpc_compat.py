"""Shared JSON-RPC helpers for MCP clients (stdio + HTTP)."""

from __future__ import annotations

from typing import Any

# Prefer current MCP revision; keep older fallback for servers that lag the spec.
MCP_PROTOCOL_VERSION_PREFERRED = "2025-03-26"
MCP_PROTOCOL_VERSION_FALLBACK = "2024-11-05"


def jsonrpc_response_id_matches(wire_id: Any, expected_id: int) -> bool:
    """True if ``wire_id`` is the JSON-RPC response id for ``expected_id`` (int or string form)."""
    if isinstance(wire_id, bool):
        return False
    if wire_id is None:
        return False
    if isinstance(wire_id, int) and wire_id == expected_id:
        return True
    if isinstance(wire_id, str):
        s = wire_id.strip()
        if s.isdigit():
            try:
                return int(s, 10) == expected_id
            except ValueError:
                return False
    return False


def mcp_initialize_params(*, protocol_version: str) -> dict[str, Any]:
    return {
        "protocolVersion": protocol_version,
        "capabilities": {},
        "clientInfo": {"name": "agent-cli", "version": "0.1"},
    }
