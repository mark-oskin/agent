"""JSON Schema tool definitions for native LLM tool-calling."""

from __future__ import annotations

import json
from typing import AbstractSet, Any, Optional

from agentlib.tools.routing import CORE_TOOL_PROMPT_DOCS

SECOND_OPINION_TOOL_ID = "second_opinion"

# Core tools with hand-maintained native schemas.
NATIVE_CORE_TOOL_IDS: frozenset[str] = frozenset(
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
        "use_git",
        "call_python",
        "run_applescript",
        "download_file",
        "tail_file",
    }
)

# Back-compat alias used in docs/tests.
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
    "use_git": {
        "type": "object",
        "properties": {
            "op": {
                "type": "string",
                "description": "Git operation: status, log, diff, add, commit, push, pull, or branch.",
            },
            "worktree": {"type": "string", "description": "Repository path (optional)."},
            "message": {"type": "string", "description": "Commit message (for commit)."},
            "remote": {"type": "string", "description": "Remote name (for push/pull)."},
            "branch": {"type": "string", "description": "Branch name (for push/pull/branch)."},
            "staged": {"type": "boolean", "description": "Staged diff only (for diff)."},
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Paths for git add.",
            },
        },
        "required": ["op"],
        "additionalProperties": False,
    },
    "call_python": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Syntactically valid Python source to execute.",
            },
            "globals": {
                "type": "object",
                "description": "Optional extra globals for exec.",
            },
        },
        "required": ["code"],
        "additionalProperties": False,
    },
    "run_applescript": {
        "type": "object",
        "properties": {
            "script": {"type": "string", "description": "AppleScript source code."},
            "timeout_ms": {"type": "integer", "description": "Timeout in milliseconds (default 20000)."},
            "echo_script": {"type": "boolean", "description": "Include script in tool output."},
            "use_temp_file": {"type": "boolean", "description": "Run via temp .applescript file."},
        },
        "required": ["script"],
        "additionalProperties": False,
    },
    "download_file": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Source URL."},
            "path": {"type": "string", "description": "Destination file path."},
        },
        "required": ["url", "path"],
        "additionalProperties": False,
    },
    "tail_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path."},
            "lines": {"type": "integer", "description": "Number of lines from end (default 20)."},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    SECOND_OPINION_TOOL_ID: {
        "type": "object",
        "properties": {
            "draft_answer": {
                "type": "string",
                "description": "Your best draft answer for the user so far.",
            },
            "rationale": {
                "type": "string",
                "description": "Why you want an independent review before finalizing.",
            },
        },
        "required": ["draft_answer", "rationale"],
        "additionalProperties": False,
    },
}


def core_tool_parameters_schema(tool_id: str) -> Optional[dict[str, Any]]:
    """Return JSON Schema for a core tool's parameters, or None."""
    return _CORE_TOOL_SCHEMAS.get(tool_id)


def _generic_object_schema(*, description: str = "") -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "additionalProperties": True}
    if description:
        schema["description"] = description
    return schema


def plugin_parameters_schema(tool_id: str) -> Optional[dict[str, Any]]:
    """Build a JSON Schema for a plugin tool from its TOOLSET metadata."""
    from agentlib.tools.plugins import PLUGIN_TOOLSETS

    tid = (tool_id or "").strip()
    if not tid:
        return None
    for ts in PLUGIN_TOOLSETS.values():
        tools = ts.get("tools") if isinstance(ts, dict) else None
        if not isinstance(tools, list):
            continue
        for td in tools:
            if not isinstance(td, dict) or str(td.get("id") or "").strip() != tid:
                continue
            custom = td.get("parameters_schema")
            if isinstance(custom, dict):
                return custom
            params = td.get("params")
            if isinstance(params, dict) and params:
                properties: dict[str, Any] = {}
                required: list[str] = []
                for key, desc in params.items():
                    ds = str(desc or "")
                    properties[str(key)] = {
                        "type": "string",
                        "description": ds or str(key),
                    }
                    if "optional" not in ds.lower():
                        required.append(str(key))
                schema: dict[str, Any] = {
                    "type": "object",
                    "properties": properties,
                    "additionalProperties": True,
                }
                if required:
                    schema["required"] = required
                return schema
            return _generic_object_schema(description=str(td.get("description") or tid))
    return None


def mcp_parameters_schema(tool_id: str) -> Optional[dict[str, Any]]:
    from agentlib.tools import mcp_registry

    schema = mcp_registry.parameters_schema(tool_id)
    if isinstance(schema, dict):
        return schema
    return None


def tool_parameters_schema(tool_id: str) -> Optional[dict[str, Any]]:
    """Native schema for any known tool id (core, plugin, MCP, second_opinion)."""
    tid = (tool_id or "").strip()
    if not tid:
        return None
    core = core_tool_parameters_schema(tid)
    if core is not None:
        return core
    plugin = plugin_parameters_schema(tid)
    if plugin is not None:
        return plugin
    return mcp_parameters_schema(tid)


def tool_has_native_schema(tool_id: str) -> bool:
    return tool_parameters_schema(tool_id) is not None


def native_tool_ids_for_enabled(
    enabled_tools: Optional[AbstractSet[str]],
    *,
    include_second_opinion: bool = False,
) -> frozenset[str]:
    """Tool ids that will be sent on the Ollama ``tools`` API for this session."""
    et = set(enabled_tools or ())
    out: set[str] = set()
    for tid in et:
        if tool_has_native_schema(tid):
            out.add(tid)
    if include_second_opinion:
        out.add(SECOND_OPINION_TOOL_ID)
    return frozenset(out)


def ollama_function_tool_definition(tool_id: str) -> Optional[dict[str, Any]]:
    """Build one OpenAI/Ollama-style tool entry for ``tool_id``."""
    params = tool_parameters_schema(tool_id)
    if params is None:
        return None
    doc = CORE_TOOL_PROMPT_DOCS.get(tool_id, tool_id)
    if tool_id == SECOND_OPINION_TOOL_ID:
        doc = (
            "second_opinion — Request an independent reviewer model before finalizing. "
            "parameters.draft_answer (string), parameters.rationale (non-empty string)."
        )
    else:
        from agentlib.tools import mcp_registry
        from agentlib.tools.plugins import plugin_tool_prompt_doc

        doc = (
            plugin_tool_prompt_doc(tool_id)
            or mcp_registry.prompt_doc(tool_id)
            or CORE_TOOL_PROMPT_DOCS.get(tool_id, tool_id)
        )
    first_line = str(doc).split("\n", 1)[0].strip()
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
    include_second_opinion: bool = False,
) -> list[dict[str, Any]]:
    """Build Ollama ``tools`` list for session-enabled tools that have native schemas."""
    native_ids = native_tool_ids_for_enabled(
        enabled_tools, include_second_opinion=include_second_opinion
    )
    out: list[dict[str, Any]] = []
    for tid in sorted(native_ids):
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


def _native_mode(*, tool_call_mode: str, primary_profile=None) -> bool:
    return tool_transport_uses_native(tool_call_mode=tool_call_mode, primary_profile=primary_profile)


def invalid_agent_response_user_content(*, tool_call_mode: str, primary_profile=None) -> str:
    """User nudge when the model response could not be parsed as answer or tool call."""
    if _native_mode(tool_call_mode=tool_call_mode, primary_profile=primary_profile):
        return (
            "Your last message was not a valid tool call or answer. "
            "To call a native tool, use the function-calling API (tool_calls), not bare parameter JSON in message text. "
            "To answer the user, reply with plain text. "
            f'To request review before finishing, call the native tool {SECOND_OPINION_TOOL_ID!r}.'
        )
    return (
        "Your last message was not valid agent JSON. "
        "Respond with JSON only and include a non-null string action. "
        'Use {"action":"tool_call","tool":<one of the allowed tools>,'
        '"parameters":{...}} or {"action":"answer","answer":"..."}.'
    )


def truncated_json_recovery_user_content(*, tool_call_mode: str, primary_profile=None) -> str:
    if _native_mode(tool_call_mode=tool_call_mode, primary_profile=primary_profile):
        return (
            "Your last response looked truncated or malformed. "
            "Reply with plain text, issue a native tool_calls request, or send one complete JSON object."
        )
    return (
        "Your last response looked like a JSON object but it was truncated/malformed "
        "(missing closing braces/quotes). Respond again with a SINGLE valid JSON object "
        "and no other text."
    )


def missing_answer_field_user_content(*, tool_call_mode: str, primary_profile=None) -> str:
    if _native_mode(tool_call_mode=tool_call_mode, primary_profile=primary_profile):
        return (
            'Your JSON had action "answer" but was missing a non-empty string field "answer". '
            "Reply with plain text, or send "
            '{"action":"answer","answer":"..."} with a non-empty answer string.'
        )
    return (
        'Your JSON had action "answer" but was missing a non-empty string field "answer". '
        "Respond again with a SINGLE valid JSON object in this exact shape:\n"
        '{"action":"answer","answer":"..."}\n'
        "No other keys, and no other text."
    )


def deliverable_full_document_user_content(*, tool_call_mode: str, primary_profile=None) -> str:
    if _native_mode(tool_call_mode=tool_call_mode, primary_profile=primary_profile):
        return (
            "Your answer is too short to be the requested multi-page document. "
            "Use read_file to load the written file, then reply with plain text containing the FULL document "
            "(the user asked for the document itself). "
            "If the file is still too short, expand it with write_file and read_file again."
        )
    return (
        "Your answer is too short to be the requested multi-page document. "
        "Use read_file to load the written file, then respond with action answer whose "
        "answer field contains the FULL document text (the user asked for the document itself). "
        "If the file is still too short, expand it with write_file and read_file again."
    )


def second_opinion_missing_rationale_user_content(*, tool_call_mode: str, primary_profile=None) -> str:
    if _native_mode(tool_call_mode=tool_call_mode, primary_profile=primary_profile):
        return (
            f"The {SECOND_OPINION_TOOL_ID} tool requires a non-empty rationale parameter. "
            f"Call {SECOND_OPINION_TOOL_ID!r} via native tool_calls with draft_answer and rationale."
        )
    return (
        f"The {SECOND_OPINION_TOOL_ID} tool requires a non-empty rationale parameter. "
        '{"action":"tool_call","tool":"second_opinion","parameters":{"draft_answer":"...","rationale":"..."}}'
    )


def second_opinion_unavailable_user_content(*, tool_call_mode: str, primary_profile=None) -> str:
    if _native_mode(tool_call_mode=tool_call_mode, primary_profile=primary_profile):
        return (
            "Second opinion is not available in this session. "
            "Reply with plain text as your final answer."
        )
    return (
        "Second opinion is not available in this session. Respond with JSON only using "
        '{"action":"answer","answer":"...","next_action":"finalize","rationale":"..."}.'
    )


def second_opinion_limit_user_content(*, tool_call_mode: str, primary_profile=None) -> str:
    if _native_mode(tool_call_mode=tool_call_mode, primary_profile=primary_profile):
        return (
            "Second opinion limit reached for this session. "
            "Reply with plain text as your final answer."
        )
    return (
        "Second opinion limit reached for this session. Respond with JSON only using "
        '{"action":"answer","answer":"...","next_action":"finalize","rationale":"..."}.'
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
