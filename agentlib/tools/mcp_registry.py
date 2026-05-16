"""Runtime registry for MCP-derived tool ids (populated by AgentApp.sync_mcp / schedule_mcp_resync)."""

from __future__ import annotations

from typing import TYPE_CHECKING, FrozenSet, Optional

if TYPE_CHECKING:
    from agentlib.mcp.cluster import McpCluster

_TOOL_IDS: FrozenSet[str] = frozenset()
_PROMPTS: dict[str, str] = {}
_UNION_ENABLED: bool = False
_CONNECT_ERRORS: list[str] = []


def clear() -> None:
    global _TOOL_IDS, _PROMPTS, _UNION_ENABLED, _CONNECT_ERRORS
    _TOOL_IDS = frozenset()
    _PROMPTS = {}
    _UNION_ENABLED = False
    _CONNECT_ERRORS = []


def install(cluster: Optional["McpCluster"], *, prefs_enabled: bool, connect_errors: Optional[list[str]] = None) -> None:
    """Replace registry contents after MCP sync."""
    global _TOOL_IDS, _PROMPTS, _UNION_ENABLED, _CONNECT_ERRORS
    _UNION_ENABLED = bool(prefs_enabled)
    if connect_errors is not None:
        _CONNECT_ERRORS = list(connect_errors)
    elif cluster is not None:
        _CONNECT_ERRORS = list(cluster.connect_errors)
    else:
        _CONNECT_ERRORS = []
    if cluster is None:
        _TOOL_IDS = frozenset()
        _PROMPTS = {}
        return
    _TOOL_IDS = frozenset(cluster.tool_index.keys())
    _PROMPTS = dict(cluster.prompt_docs)


def all_ids() -> FrozenSet[str]:
    return _TOOL_IDS


def global_mcp_prefs_enabled() -> bool:
    """True when the last ``install()`` had MCP enabled in prefs (servers may be connected)."""
    return _UNION_ENABLED


def union_into_session_enabled() -> bool:
    """Deprecated alias for :func:`global_mcp_prefs_enabled` (name was misleading)."""
    return global_mcp_prefs_enabled()


def prompt_doc(tool_id: str) -> str:
    return str(_PROMPTS.get(str(tool_id or "").strip(), "") or "")


def is_mcp_tool(tool_id: str) -> bool:
    return str(tool_id or "").strip() in _TOOL_IDS


def last_connect_errors() -> list[str]:
    return list(_CONNECT_ERRORS)
