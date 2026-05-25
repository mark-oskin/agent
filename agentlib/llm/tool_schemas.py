"""JSON Schema tool definitions for native LLM tool-calling (Phase 1: core Ollama tools)."""

from __future__ import annotations

from typing import AbstractSet, Any, Optional

from agentlib.tools.routing import CORE_TOOL_PROMPT_DOCS

# Phase 1: hand-maintained schemas for a small core set (expand in later phases).
NATIVE_PHASE1_TOOL_IDS: frozenset[str] = frozenset(
    {
        "search_web",
        "search_web_fetch_top",
        "read_file",
        "run_command",
        "fetch_page",
        "write_file",
        "grep",
        "list_directory",
        "replace_text",
    }
)

_CORE_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "search_web": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Web search terms (non-empty).",
            },
            "max_results": {
                "type": "integer",
                "description": "Number of result rows to parse (1–30).",
                "minimum": 1,
                "maximum": 30,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "search_web_fetch_top": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Web search terms (non-empty).",
            },
            "max_results": {
                "type": "integer",
                "description": "Number of result rows to parse (1–30).",
                "minimum": 1,
                "maximum": 30,
            },
            "fetch_top_n": {
                "type": "integer",
                "description": "Number of top result pages to fetch excerpts from (1–10).",
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "read_file": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to read.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    "run_command": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to run.",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    },
    "fetch_page": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Single http(s) URL to fetch.",
            },
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Multiple http(s) URLs to fetch (batch limit applies).",
            },
        },
        "additionalProperties": False,
    },
    "write_file": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to write.",
            },
            "content": {
                "type": "string",
                "description": "Full file contents to write.",
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
    "grep": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Non-empty Python re regex to search for.",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search (default .).",
            },
            "glob_pattern": {
                "type": "string",
                "description": "Glob filter when searching a directory (e.g. *.py).",
            },
            "max_matches": {
                "type": "integer",
                "description": "Maximum matching lines to return (default 200, max 5000).",
                "minimum": 1,
                "maximum": 5000,
            },
            "max_files": {
                "type": "integer",
                "description": "Maximum files to scan in a directory search (default 8000).",
                "minimum": 1,
            },
            "ignore_case": {
                "type": "boolean",
                "description": "Case-insensitive regex match.",
            },
        },
        "required": ["pattern"],
        "additionalProperties": False,
    },
    "list_directory": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    "replace_text": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to edit.",
            },
            "pattern": {
                "type": "string",
                "description": "Python re regex to find.",
            },
            "replacement": {
                "type": "string",
                "description": "Replacement text (may use regex backrefs).",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all matches (default true).",
            },
        },
        "required": ["path", "pattern", "replacement"],
        "additionalProperties": False,
    },
}


def core_tool_parameters_schema(tool_id: str) -> Optional[dict[str, Any]]:
    """Return JSON Schema for a Phase 1 core tool's parameters, or None."""
    return _CORE_TOOL_SCHEMAS.get(tool_id)


def ollama_function_tool_definition(tool_id: str) -> Optional[dict[str, Any]]:
    """Build one OpenAI/Ollama-style tool entry for ``tool_id``."""
    params = core_tool_parameters_schema(tool_id)
    if params is None:
        return None
    doc = CORE_TOOL_PROMPT_DOCS.get(tool_id, tool_id)
    first_line = doc.split("\n", 1)[0].strip()
    description = first_line if first_line else tool_id
    return {
        "type": "function",
        "function": {
            "name": tool_id,
            "description": description,
            "parameters": params,
        },
    }


def ollama_tools_for_enabled(
    enabled_tools: Optional[AbstractSet[str]],
    *,
    phase1_only: bool = True,
) -> list[dict[str, Any]]:
    """
    Build Ollama ``tools`` list for session-enabled tools that have native schemas.

    When ``phase1_only`` is True, only ``NATIVE_PHASE1_TOOL_IDS`` are included.
    """
    et = set(enabled_tools or ())
    if not et:
        return []
    allowed = NATIVE_PHASE1_TOOL_IDS if phase1_only else frozenset(_CORE_TOOL_SCHEMAS)
    out: list[dict[str, Any]] = []
    for tid in sorted(allowed):
        if tid not in et:
            continue
        entry = ollama_function_tool_definition(tid)
        if entry is not None:
            out.append(entry)
    return out


def first_tool_call_name(tool_calls) -> Optional[str]:
    """Best-effort tool name from Ollama/OpenAI ``tool_calls`` chunks."""
    if not tool_calls:
        return None
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        if not isinstance(fn, dict):
            continue
        name = (fn.get("name") or "").strip()
        if name:
            return name
    return None


def tool_transport_label(tool_id: str, transport: str) -> str:
    """Short tag for logs, e.g. ``[native] search_web``."""
    tid = (tool_id or "?").strip() or "?"
    kind = (transport or "json").strip().lower()
    return f"[{kind}] {tid}"


def preparing_tool_progress_line(*, tool_calls, content: str) -> str:
    """
    User-visible progress while the model is selecting a tool.

    ``tool_calls`` (API field) → native transport; JSON in ``content`` → json transport.
    """
    name = first_tool_call_name(tool_calls)
    if name:
        return f"Preparing native tool call ({name})…"
    if content and '"action"' in content and "tool_call" in content:
        return "Preparing JSON tool call (message content)…"
    return "Preparing tool call…"


def tool_transport_uses_native(*, tool_call_mode: str, primary_profile=None) -> bool:
    """True when agent turns should use Ollama native ``tools`` (not JSON-in-content)."""
    from agentlib.prompts import use_native_tool_prompt

    return use_native_tool_prompt(tool_call_mode=tool_call_mode, primary_profile=primary_profile)


def tool_call_only_nudge(*, tool_call_mode: str, primary_profile=None) -> str:
    """Short suffix for user_info gates when the model must call a tool instead of answering."""
    if tool_transport_uses_native(tool_call_mode=tool_call_mode, primary_profile=primary_profile):
        return (
            "Call the required tool using the native function-calling API (tool_calls), "
            "not JSON {\"action\":\"tool_call\",...} text in message content."
        )
    return "Respond with JSON tool_call only."


def invalid_agent_response_user_content(*, tool_call_mode: str, primary_profile=None) -> str:
    """User nudge when the model response could not be parsed as answer or tool call."""
    if tool_transport_uses_native(tool_call_mode=tool_call_mode, primary_profile=primary_profile):
        return (
            "Your last message was not a valid tool call or answer. "
            "To call a native tool, use the function-calling API (tool_calls), not bare parameter JSON in message text. "
            "To answer the user, reply with plain text, or use "
            '{"action":"answer","answer":"..."} when you need structured fields like next_action. '
            'For JSON-only tools, use {"action":"tool_call","tool":<name>,"parameters":{...}}.'
        )
    return (
        "Your last message was not valid agent JSON. "
        "Respond with JSON only and include a non-null string action. "
        'Use {"action":"tool_call","tool":<one of the allowed tools>,'
        '"parameters":{...}} or {"action":"answer","answer":"..."}.'
    )


def web_search_required_user_content(
    tool_name: str,
    suggested_query: str,
    *,
    tool_call_mode: str,
    primary_profile=None,
    lead_in: str = "Before answering",
) -> str:
    """User message injected when web verification is required before answering."""
    q = (suggested_query or "").strip()
    tool = (tool_name or "search_web").strip()
    if tool_transport_uses_native(tool_call_mode=tool_call_mode, primary_profile=primary_profile):
        return (
            f"{lead_in}, you MUST call the native tool {tool!r} (function calling API).\n"
            f"Do not emit JSON {{\"action\":\"tool_call\",...}} text for this tool.\n"
            f'Suggested query parameter: "{q}"'
        )
    return (
        f"{lead_in}, you MUST call the tool {tool}.\n"
        "Respond with JSON only in tool_call form.\n"
        f'Suggested query: "{q}"'
    )
