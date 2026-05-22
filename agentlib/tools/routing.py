from __future__ import annotations

import difflib
import re
from typing import AbstractSet, Optional

from . import mcp_registry
from .plugins import (
    PLUGIN_TOOL_HANDLERS,
    PLUGIN_TOOLSETS,
    PLUGIN_TOOLSET_TRIGGERS,
    PLUGIN_TOOL_TO_TOOLSET,
    plugin_tool_entries,
    plugin_tools_for_toolset,
)


_WEB_SEARCH_TOOLS_PRIORITY: tuple[str, ...] = ("search_web", "search_web_fetch_top")


def preferred_web_search_tool(enabled_tools: Optional[AbstractSet[str]]) -> Optional[str]:
    """Prefer ``search_web`` when enabled; otherwise ``search_web_fetch_top`` if enabled."""
    et = set(enabled_tools or ())
    for name in _WEB_SEARCH_TOOLS_PRIORITY:
        if name in et:
            return name
    return None


CORE_TOOL_ENTRIES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("search_web", "Web search", ("web", "search", "web search", "internet search")),
    (
        "search_web_fetch_top",
        "Web search + fetch top results",
        ("search+fetch", "search fetch", "web verify", "web verify search"),
    ),
    ("fetch_page", "Fetch a web page", ("fetch", "download page", "get url")),
    ("run_command", "Run a shell command", ("shell", "bash", "run", "exec")),
    ("use_git", "Run a git operation", ("git", "git status", "git diff")),
    (
        "write_file",
        "Write or overwrite a file",
        ("write", "save file", "create file", "file write"),
    ),
    ("read_file", "Read a file", ("read", "cat", "open file", "slurp")),
    (
        "grep",
        "Search files with a regex (ripgrep-like, Python re)",
        ("rg", "ripgrep", "code search", "search in repo", "find in files"),
    ),
    (
        "list_directory",
        "List directory contents",
        ("ls", "dir", "list dir", "folder", "directory"),
    ),
    ("download_file", "Download a file from URL", ("download", "wget", "curl download")),
    ("tail_file", "Read end of a file (tail)", ("tail", "log tail")),
    (
        "replace_text",
        "Search-and-replace in a file",
        ("replace", "search replace", "sed"),
    ),
    ("call_python", "Run Python code in-process", ("python", "py", "eval", "code eval")),
)


# Single source of truth for tool-call prompt docs (core tools).
# These strings are rendered into the system prompt under "Parameters per tool".
CORE_TOOL_PROMPT_DOCS: dict[str, str] = {
    "search_web": (
        "search_web — parameters.query (non-empty string, the web search terms); optional parameters.max_results "
        "(integer 1–30, how many result rows to parse; default from AGENT_SEARCH_WEB_MAX_RESULTS, else 5). "
        "Backend is prefs agent.search_web_backend: ddg (default), searxng (needs agent.searxng_url), or brave "
        "(needs agent.brave_search_api_key)."
    ),
    "search_web_fetch_top": (
        "search_web_fetch_top — parameters.query (non-empty string); optional parameters.max_results (1–30) and "
        "parameters.fetch_top_n (1–10, default 10). Returns web results plus fetched excerpts."
    ),
    "fetch_page": (
        "fetch_page — parameters.url (string) and/or parameters.urls (array of http/https URLs); "
        "multiple URLs return combined text (batch limit applies)."
    ),
    "run_command": "run_command — parameters.command (string, shell command to run).",
    "use_git": (
        "use_git — parameters.op (string: status|log|diff|add|commit|push|pull|branch), "
        "optional parameters.worktree (repo path), parameters.message (for commit), "
        "parameters.remote / parameters.branch (for push/pull), parameters.staged (boolean, for diff), "
        "parameters.paths (array of strings for add)."
    ),
    "write_file": "write_file — parameters.path (file path string), parameters.content (string to write).",
    "read_file": "read_file — parameters.path (file path string).",
    "grep": (
        "grep — parameters.pattern (non-empty Python ``re`` regex string); optional parameters.path "
        "(file or directory, default ``.``); optional parameters.glob_pattern (e.g. ``*.py``) when searching a directory; "
        "optional parameters.max_matches (int, default 200, max 5000); optional parameters.max_files (int, default 8000); "
        "optional parameters.ignore_case (boolean). Returns matching lines as ``path:line:text`` plus a short summary footer."
    ),
    "list_directory": "list_directory — parameters.path (directory path string).",
    "download_file": "download_file — parameters.url (source URL string), parameters.path (destination file path).",
    "tail_file": (
        "tail_file — parameters.path (file path string); optional: parameters.lines (integer, default 20)."
    ),
    "replace_text": (
        "replace_text — parameters.path, parameters.pattern (regex string), parameters.replacement (string); "
        "optional: parameters.replace_all (boolean, default true). "
        "If the regex matches nowhere, the file is left unchanged and the tool explains why (indentation, "
        "\\n vs spaces, unescaped . ( ) + * etc.)."
    ),
    "call_python": (
        "call_python — parameters.code (string, syntactically valid Python ONLY). "
        "Tool output includes STDOUT from print() (if any) plus a JSON summary of assigned variables (locals); "
        "use print for human-readable trace. "
        "Never put shell/batch/cmd text, pseudo-code, or natural-language document drafts in code; "
        "those belong in write_file content or in action answer. "
        "Optional: parameters.globals (object, extra globals)."
    ),
}


def core_tool_prompt_doc(tool_id: str) -> str:
    """Prompt-doc line for a core tool, or empty string if unknown."""
    return str(CORE_TOOL_PROMPT_DOCS.get(str(tool_id or "").strip(), "") or "")


TOOL_ALIASES: dict[str, str] = {}


def canonicalize_user_tool_phrase(phrase: str) -> str:
    s = (phrase or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = s.replace("-", "_")
    return s


def all_known_tools() -> frozenset[str]:
    return frozenset(
        {tid for tid, _label, _aliases in CORE_TOOL_ENTRIES}
        | set(PLUGIN_TOOL_HANDLERS.keys())
        | set(mcp_registry.all_ids())
    )


def register_tool_aliases() -> None:
    TOOL_ALIASES.clear()
    for internal, _label, aliases in (*CORE_TOOL_ENTRIES, *plugin_tool_entries()):
        for phrase in (internal, *aliases):
            key = canonicalize_user_tool_phrase(phrase)
            if key:
                TOOL_ALIASES[key] = internal


def resolve_tool_token(phrase: str) -> Optional[str]:
    c = canonicalize_user_tool_phrase(phrase)
    if not c:
        return None
    if c in all_known_tools():
        return c
    return TOOL_ALIASES.get(c)


def normalize_tool_name(token: str) -> Optional[str]:
    return resolve_tool_token(token)


def _all_tool_name_suggestion_pool() -> list[str]:
    pool = set(all_known_tools())
    pool.update(TOOL_ALIASES.keys())
    pool.add("second_opinion")
    return sorted(pool)


def format_unknown_tool_hint(phrase: str) -> str:
    c = canonicalize_user_tool_phrase(phrase)
    lines = [f"Unknown tool {phrase!r}."]
    if c:
        near = difflib.get_close_matches(c, _all_tool_name_suggestion_pool(), n=4, cutoff=0.55)
        if near:
            bits = []
            for m in near:
                if m == "second_opinion":
                    bits.append("second_opinion (feature, not a tool)")
                    continue
                internal = resolve_tool_token(m) or m
                bits.append(f"{m} → {internal}" if m != internal else internal)
            lines.append("Did you mean: " + ", ".join(bits) + "?")
    lines.append("Run /set tools (or --list-tools) for every tool and its id.")
    return "\n".join(lines)


def format_settings_tools_list(enabled_tools: AbstractSet[str]) -> str:
    lines = ["Core tools for this session (id in parentheses, use either):"]
    for internal, label, _aliases in CORE_TOOL_ENTRIES:
        on = "on" if internal in enabled_tools else "off"
        lines.append(f"  [{on}] {label}  ({internal})")
    mcp_ids = sorted(mcp_registry.all_ids())
    if mcp_ids:
        lines.append("")
        lines.append("MCP tools (this process; enable per session with /mcp session on):")
        for tid in mcp_ids:
            on = "on" if tid in enabled_tools else "off"
            lines.append(f"  [{on}] {tid}")
    lines.append(
        "Enable/disable: /set tools <id or phrase> enable|disable  "
        "(e.g. /set tools web search enable)  or  -enable_tool shell"
    )
    lines.append(
        "Note: /skill or agent.skill_auto_match_triggers may narrow tools per message; "
        "use /settings verbose 1 to see effective tools for a turn."
    )
    return "\n".join(lines)


def route_active_toolsets_for_request(user_query: str, enabled_toolsets: AbstractSet[str]) -> set[str]:
    ets = {str(x).strip().lower() for x in (enabled_toolsets or set()) if str(x).strip()}
    if not ets:
        return set()
    if len(ets) == 1:
        return set(ets)
    q = (user_query or "").strip().lower()
    if not q:
        return set(ets)
    active: set[str] = set()
    for ts in sorted(ets):
        tr = PLUGIN_TOOLSET_TRIGGERS.get(ts) or []
        for one in tr:
            s = str(one).strip()
            if not s:
                continue
            if s.startswith("regex:"):
                pat = s[len("regex:") :].strip()
                if pat:
                    try:
                        if re.search(pat, q, flags=re.I):
                            active.add(ts)
                            break
                    except re.error:
                        continue
            else:
                if s.lower() in q:
                    active.add(ts)
                    break
    return active or set(ets)


def effective_enabled_tools_for_turn(
    *,
    base_enabled_tools: AbstractSet[str],
    enabled_toolsets: AbstractSet[str],
    user_query: str,
) -> frozenset[str]:
    base = set(base_enabled_tools or set())
    active_ts = route_active_toolsets_for_request(user_query, enabled_toolsets)
    for ts in active_ts:
        base.update(plugin_tools_for_toolset(ts))
    # MCP tools are per-session via ``enabled_tools`` (not auto-unioned). Global ``agent.mcp_enabled``
    # only starts shared subprocesses; each session opts in with ``/mcp session on`` or ``/set tools``.
    base = base & set(all_known_tools())
    return frozenset(base)


def tool_policy_runner_text(enabled_tools: Optional[AbstractSet[str]]) -> str:
    e = set(all_known_tools()) if enabled_tools is None else set(enabled_tools)
    if e == set(all_known_tools()):
        return ""
    disabled = sorted(all_known_tools() - e)
    allowed = sorted(e & all_known_tools())
    return (
        "Runner: tool policy — you MUST NOT use tool_call for: "
        + ", ".join(disabled)
        + ". Only these tools may be invoked (you have explicit permission to use them): "
        + ", ".join(allowed)
        + "."
    )


def describe_tool_call_contract(tool_id: str) -> str:
    tid = (tool_id or "").strip()
    if not tid:
        return "Unknown tool."
    if mcp_registry.is_mcp_tool(tid):
        doc = mcp_registry.prompt_doc(tid)
        out = [f"Tool: {tid} (MCP)"]
        if doc:
            out.append(doc)
        else:
            out.append("Contract: parameters JSON object per MCP inputSchema for this tool.")
        out.append("Returns: string (tool output)")
        return "\n".join(out)
    if tid in PLUGIN_TOOL_HANDLERS:
        ts = PLUGIN_TOOL_TO_TOOLSET.get(tid) or ""
        rec = PLUGIN_TOOLSETS.get(ts) or {}
        tools = rec.get("tools") if isinstance(rec, dict) else None
        td = None
        if isinstance(tools, list):
            for one in tools:
                if isinstance(one, dict) and str(one.get("id") or "").strip() == tid:
                    td = one
                    break
        desc = str((td or {}).get("description") or "").strip() if isinstance(td, dict) else ""
        aliases = (td or {}).get("aliases") if isinstance(td, dict) else None
        if not isinstance(aliases, (list, tuple)):
            aliases = ()
        params = (td or {}).get("params") if isinstance(td, dict) else None
        returns = str((td or {}).get("returns") or "").strip() if isinstance(td, dict) else ""
        out = [f"Tool: {tid} (plugin toolset {ts!r})"]
        if desc:
            out.append(f"Description: {desc}")
        if aliases:
            out.append("Aliases: " + ", ".join(str(a) for a in aliases))
        if isinstance(params, dict) and params:
            out.append("Parameters:")
            for k, v in params.items():
                out.append(f"  - {k}: {str(v).strip()}")
        else:
            out.append("Parameters: (tool-specific; accepts a JSON object in parameters)")
        out.append("Returns: " + (returns if returns else "string (tool output)"))
        return "\n".join(out)

    core = {
        "search_web": "parameters.query (string); optional max_results (1–30). Returns: formatted web results string.",
        "search_web_fetch_top": "parameters.query (string); optional max_results (1–30), fetch_top_n (1–10). Returns: web results plus fetched excerpts (string).",
        "fetch_page": "parameters.url (http/https URL) and/or parameters.urls (array). Returns: fetched page text (PDFs as extracted text when enabled).",
        "run_command": "parameters.command (shell command). Returns: STDOUT/STDERR string.",
        "use_git": "parameters.op plus additional fields (status|log|diff|add|commit|push|pull|branch). Returns: git output string.",
        "write_file": "parameters.path, parameters.content. Returns: success/error string.",
        "read_file": "parameters.path. Returns: file contents string (or error).",
        "grep": "parameters.pattern; optional path, glob_pattern, max_matches, max_files, ignore_case. Returns: path:line:text lines + summary (or error).",
        "list_directory": "parameters.path. Returns: JSON-ish list string (or error).",
        "download_file": "parameters.url, parameters.path. Returns: success/error string.",
        "tail_file": "parameters.path, optional lines. Returns: tail text string.",
        "replace_text": "parameters.path, pattern, replacement, optional replace_all. Returns: success/error string.",
        "call_python": "parameters.code, optional globals. Returns: stdout + locals JSON string (or error).",
    }
    if tid in core:
        return f"Tool: {tid} (core)\nContract: {core[tid]}"
    return "Unknown tool."

