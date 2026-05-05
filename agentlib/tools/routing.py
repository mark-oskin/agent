from __future__ import annotations

import difflib
import re
from typing import AbstractSet, Optional

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


TOOL_ALIASES: dict[str, str] = {}


def canonicalize_user_tool_phrase(phrase: str) -> str:
    s = (phrase or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = s.replace("-", "_")
    return s


def all_known_tools() -> frozenset[str]:
    return frozenset({tid for tid, _label, _aliases in CORE_TOOL_ENTRIES} | set(PLUGIN_TOOL_HANDLERS.keys()))


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
    lines.append(
        "You can use plain phrases, e.g. /set disable web search  "
        "or  -disable_tool shell"
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
        "fetch_page": "parameters.url (http/https URL). Returns: fetched page text (string).",
        "run_command": "parameters.command (shell command). Returns: STDOUT/STDERR string.",
        "use_git": "parameters.op plus additional fields (status|log|diff|add|commit|push|pull|branch). Returns: git output string.",
        "write_file": "parameters.path, parameters.content. Returns: success/error string.",
        "read_file": "parameters.path. Returns: file contents string (or error).",
        "list_directory": "parameters.path. Returns: JSON-ish list string (or error).",
        "download_file": "parameters.url, parameters.path. Returns: success/error string.",
        "tail_file": "parameters.path, optional lines. Returns: tail text string.",
        "replace_text": "parameters.path, pattern, replacement, optional replace_all. Returns: success/error string.",
        "call_python": "parameters.code, optional globals. Returns: stdout + locals JSON string (or error).",
    }
    if tid in core:
        return f"Tool: {tid} (core)\nContract: {core[tid]}"
    return "Unknown tool."

