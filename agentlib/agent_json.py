"""Parse model output into agent JSON dicts (tool_call / answer / router shapes)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import AbstractSet, Callable, FrozenSet, Optional

_AGENT_TOP_LEVEL_PARAM_KEYS = frozenset(
    {
        "query",
        "url",
        "command",
        "path",
        "content",
        "lines",
        "pattern",
        "replacement",
        "replace_all",
        "code",
        "globals",
        "op",
        "operation",
        "worktree",
        "message",
        "remote",
        "branch",
        "staged",
        "paths",
        "m",
        "n",
        "max_results",
        "max",
        "num_results",
        "limit",
    }
)


@dataclass(frozen=True)
class AgentJsonDeps:
    """Callbacks supplied by the host (agent.py wiring) for tool sets and param normalization."""

    all_known_tools: Callable[[], FrozenSet[str]]
    coerce_enabled_tools: Callable[[Optional[AbstractSet[str]]], AbstractSet[str]]
    merge_tool_param_aliases: Callable[[str, dict], dict]


def clean_json_response(resp_text):
    if resp_text is None:
        return ""
    try:
        start = resp_text.index("{")
        return resp_text[start:]
    except Exception:
        return resp_text


def iter_balanced_brace_objects(text: str):
    """Yield `{...}` spans with brace depth matching outside of JSON strings (double-quoted)."""
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, c in enumerate(text):
        if escape:
            escape = False
            continue
        if in_str:
            if c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start : i + 1]
                    start = -1


def normalize_unicode_json_quotes(s: str) -> str:
    """Map common Unicode “smart” double quotes to ASCII so delimiters parse as JSON."""
    if not s:
        return s
    trans = str.maketrans(
        {
            "\u201c": '"',
            "\u201d": '"',
            "\u201e": '"',
            "\u2033": '"',
            "\uff02": '"',
        }
    )
    return s.translate(trans)


def escape_controls_inside_json_strings(s: str) -> str:
    """
    RFC 8259 forbids literal ASCII control characters inside JSON strings.
    Many LLMs emit literal newlines/tabs inside \"answer\" values anyway — escape them so json.loads succeeds.
    Tracks double-quoted regions using the same rules as iter_balanced_brace_objects.
    """
    if not s:
        return s
    out: list[str] = []
    in_str = False
    escape = False
    for c in s:
        if escape:
            escape = False
            out.append(c)
            continue
        if in_str:
            if c == "\\":
                escape = True
                out.append(c)
                continue
            if c == '"':
                in_str = False
                out.append(c)
                continue
            o = ord(c)
            if o < 32:
                if c == "\n":
                    out.append("\\n")
                elif c == "\r":
                    out.append("\\r")
                elif c == "\t":
                    out.append("\\t")
                elif c == "\f":
                    out.append("\\f")
                elif c == "\b":
                    out.append("\\b")
                else:
                    out.append("\\u%04x" % o)
                continue
            out.append(c)
            continue
        if c == '"':
            in_str = True
        out.append(c)
    return "".join(out)


def fallback_extract_answer_field(raw: str) -> Optional[str]:
    """
    Last resort when JSON is too broken to parse: scan for \"answer\"\\s*:\\s*\" and read a
    double-quoted string with standard JSON-style escapes (handles embedded quotes and newlines badly; good enough).
    """
    text = normalize_unicode_json_quotes(raw.strip())
    # Avoid scraping unrelated JSON fragments that mention "answer" as a word or nested keys.
    if not re.search(r'"action"\s*:\s*"answer"', text):
        return None
    m = re.search(r'"answer"\s*:\s*"', text)
    if not m:
        return None
    i = m.end()
    buf: list[str] = []
    escape = False
    while i < len(text):
        c = text[i]
        if escape:
            escape = False
            # Map common JSON escapes back to literal text for the answer we return.
            if c == "n":
                buf.append("\n")
            elif c == "r":
                buf.append("\r")
            elif c == "t":
                buf.append("\t")
            elif c == '"':
                buf.append('"')
            elif c == "\\":
                buf.append("\\")
            elif c == "/":
                buf.append("/")
            elif c == "f":
                buf.append("\f")
            elif c == "b":
                buf.append("\b")
            elif c == "u" and i + 4 < len(text):
                hex_part = text[i + 1 : i + 5]
                if len(hex_part) == 4 and all(ch in "0123456789abcdefABCDEF" for ch in hex_part):
                    buf.append(chr(int(hex_part, 16)))
                    i += 4
                else:
                    buf.append("\\")
                    buf.append(c)
            else:
                buf.append("\\")
                buf.append(c)
            i += 1
            continue
        if c == "\\":
            escape = True
            i += 1
            continue
        if c == '"':
            return "".join(buf)
        buf.append(c)
        i += 1
    return None


def try_json_loads_object(s: str):
    """Parse a JSON object string; apply light repairs for common model mistakes."""
    if not s or not s.strip():
        return None
    s = s.strip()

    variants = [
        s,
        normalize_unicode_json_quotes(s),
        escape_controls_inside_json_strings(s),
        escape_controls_inside_json_strings(normalize_unicode_json_quotes(s)),
    ]
    seen = set()
    uniq_variants = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            uniq_variants.append(v)

    for cand in uniq_variants:
        fixes = (
            cand,
            re.sub(r",\s*}", "}", cand),
            re.sub(r",\s*]", "]", cand),
        )
        for fix in fixes:
            try:
                v = json.loads(fix)
                if isinstance(v, dict):
                    return v
            except json.JSONDecodeError:
                continue

    extracted = fallback_extract_answer_field(normalize_unicode_json_quotes(s))
    if extracted is not None and extracted.strip():
        return {"action": "answer", "answer": extracted}

    return None


def best_agent_dict_from_text(text: str, known_tools: FrozenSet[str]) -> Optional[dict]:
    """
    Find the best JSON object in mixed prose: prefer dicts with 'action',
    then dicts whose tool name is known.
    """
    if not text or not text.strip():
        return None
    candidates = []
    for span in iter_balanced_brace_objects(text):
        parsed = try_json_loads_object(span)
        if isinstance(parsed, dict):
            candidates.append(parsed)
    if not candidates:
        return None

    def score(d: dict) -> tuple:
        action = d.get("action")
        tool = d.get("tool") or (action if action in known_tools else None)
        has_action = 1 if action else 0
        known_tool = 1 if tool in known_tools else 0
        tool_call_shape = 1 if action == "tool_call" and tool in known_tools else 0
        return (tool_call_shape, known_tool, has_action)

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def extract_json_object_from_text(text: str, deps: AgentJsonDeps) -> Optional[str]:
    """If the model buried JSON in prose/thinking, pull out the best object span."""
    best = best_agent_dict_from_text(text, deps.all_known_tools())
    if not best:
        return None
    try:
        return json.dumps(best)
    except (TypeError, ValueError):
        return None


def parse_tool_arguments(arguments):
    """Normalize tool `arguments` from Ollama (str, dict, or malformed JSON)."""
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return dict(arguments)
    if isinstance(arguments, str):
        s = arguments.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass
        # Trailing commas / truncated fragments
        for fix in (s, re.sub(r",\s*}", "}", s), re.sub(r",\s*]", "]", s)):
            try:
                parsed = json.loads(fix)
                return dict(parsed) if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                continue
        # Single quotes used like Python dict (best-effort)
        if "'" in s and '"' not in s:
            try:
                parsed = json.loads(s.replace("'", '"'))
                return dict(parsed) if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                pass
    return {}


def tool_call_to_agent_dict(function_name: str, arguments):
    """Map Ollama native tool_calls to our agent JSON shape."""
    raw = parse_tool_arguments(arguments)
    if not raw:
        raw = {}
    if raw.get("action") == "tool_call" and raw.get("tool"):
        return {
            "action": "tool_call",
            "tool": raw["tool"],
            "parameters": raw.get("parameters") or {},
        }
    name = (function_name or "").strip()
    if name == "tool_call" and raw.get("tool"):
        return {
            "action": "tool_call",
            "tool": raw["tool"],
            "parameters": raw.get("parameters") or {},
        }
    params = dict(raw)
    if name.startswith("tool."):
        tool = name.split(".", 1)[1]
    elif name.startswith("functions.") or name.startswith("function."):
        tool = name.split(".", 1)[1]
    else:
        tool = name or "unknown"
    if "filename" in params and "path" not in params:
        params["path"] = params.pop("filename")
    return {"action": "tool_call", "tool": tool, "parameters": params}


def tool_calls_to_agent_json_text(
    tool_calls, enabled_tools: Optional[AbstractSet[str]], deps: AgentJsonDeps
) -> Optional[str]:
    """Pick the first native tool_call that maps to a known, session-enabled tool."""
    et = deps.coerce_enabled_tools(enabled_tools)
    if not tool_calls:
        return None
    known = deps.all_known_tools()
    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        args = fn.get("arguments")
        mapped = tool_call_to_agent_dict(name, args)
        t = mapped.get("tool") if mapped else None
        if mapped and t in known and t in et:
            return json.dumps(mapped)
    return None


def message_to_agent_json_text(
    msg: dict, enabled_tools: Optional[AbstractSet[str]], deps: AgentJsonDeps
) -> str:
    """Build a JSON string for parse_agent_json from an Ollama chat message."""
    # Prefer structured native tool_calls when present (works even if content is noise).
    from_tools = tool_calls_to_agent_json_text(msg.get("tool_calls"), enabled_tools, deps)
    if from_tools:
        return from_tools

    text = (msg.get("content") or "").strip()
    if text:
        extracted = extract_json_object_from_text(text, deps)
        if extracted:
            return extracted
        best = best_agent_dict_from_text(text, deps.all_known_tools())
        if best:
            try:
                return json.dumps(best)
            except (TypeError, ValueError):
                pass
        return text

    thinking = (msg.get("thinking") or "").strip()
    if thinking:
        extracted = extract_json_object_from_text(thinking, deps)
        if extracted:
            return extracted
        best = best_agent_dict_from_text(thinking, deps.all_known_tools())
        if best:
            try:
                return json.dumps(best)
            except (TypeError, ValueError):
                pass
        return thinking
    return ""


def _is_tool_call_intent(out: dict, known_tools: FrozenSet[str]) -> bool:
    a = out.get("action")
    if a == "tool_call":
        return True
    if a in known_tools:
        return True
    if out.get("tool") in known_tools:
        return True
    return False


def normalize_agent_dict(d: dict, deps: AgentJsonDeps) -> dict:
    """
    Coerce common alternate shapes into what main() expects:
    - tool name as action, args at top level, tool_name vs tool, etc.
    """
    known_tools = deps.all_known_tools()
    if not isinstance(d, dict):
        return {"action": "answer", "answer": str(d)}
    out = dict(d)
    action = out.get("action")
    if isinstance(action, str):
        action = action.strip()
        out["action"] = action
    elif action is None or action is False:
        out.pop("action", None)
        action = None
    else:
        # Models sometimes emit JSON null / numbers for action; treat as missing.
        out.pop("action", None)
        action = None

    # answer synonyms (avoid "content" — it is also a tool parameter name)
    if out.get("answer") is None:
        for k in ("response", "message", "text"):
            if k in out and out[k] is not None and isinstance(out[k], str):
                out["answer"] = out[k]
                break
        # Some models use {"content": "..."} as the final answer without action.
        if out.get("answer") is None and isinstance(out.get("content"), str) and out["content"].strip():
            out["answer"] = out["content"]

    # tool name aliases
    if out.get("tool") is None:
        for alias in ("tool_name", "toolName", "function_name", "function"):
            v = out.get(alias)
            if isinstance(v, str) and v in known_tools:
                out["tool"] = v
                break
        if out.get("tool") is None and isinstance(out.get("name"), str) and out["name"] in known_tools:
            out["tool"] = out["name"]

    # Infer missing action after aliases / answer fields are filled in.
    if not action or (isinstance(action, str) and action.lower() in ("null", "none", "")):
        if out.get("tool") in known_tools:
            out["action"] = "tool_call"
            action = "tool_call"
        elif out.get("answer") is not None and isinstance(out.get("answer"), str) and out["answer"].strip():
            out["action"] = "answer"
            action = "answer"
            # If we promoted content -> answer, drop content to avoid ambiguity with write_file's content param.
            if "content" in out and out.get("answer") == out.get("content"):
                out.pop("content", None)

    # {"action": "run_command", "command": "..."}  (action is the tool id)
    if action in known_tools and out.get("tool") is None:
        out["tool"] = action
        out["action"] = "tool_call"

    tool_intent = _is_tool_call_intent(out, known_tools)
    params = out.get("parameters") if tool_intent else {}
    if not isinstance(params, dict):
        params = {}
    if tool_intent:
        for alias in ("args", "arguments", "params"):
            alt = out.get(alias)
            if isinstance(alt, dict):
                params = {**alt, **params}
                out.pop(alias, None)

        for k in list(out.keys()):
            if k in _AGENT_TOP_LEVEL_PARAM_KEYS:
                params.setdefault(k, out[k])

        tool_name = out.get("tool")
        if tool_name == "search_web":
            for alt in ("q", "search", "keywords", "keyword"):
                if alt in out and out[alt] is not None:
                    params.setdefault("query", out[alt])
        elif tool_name == "fetch_page":
            for alt in ("href", "link", "uri"):
                if alt in out and out[alt] is not None:
                    params.setdefault("url", out[alt])
        elif tool_name == "run_command":
            for alt in ("cmd", "shell", "line"):
                if alt in out and out[alt] is not None:
                    params.setdefault("command", out[alt])
        elif tool_name == "use_git":
            for alt in ("operation", "git_op", "subcommand", "sub_cmd"):
                if alt in out and out[alt] is not None:
                    params.setdefault("op", out[alt])
            for alt in ("cwd", "repo", "work_tree"):
                if alt in out and out[alt] is not None:
                    params.setdefault("worktree", out[alt])
            for alt in ("files", "file", "file_paths"):
                if alt in out and out[alt] is not None and "paths" not in params:
                    params.setdefault("paths", out[alt])

        merge = deps.merge_tool_param_aliases
        params = merge(tool_name or "", params)

        out["parameters"] = params

        _extra_top = frozenset(
            {
                "q",
                "search",
                "keywords",
                "keyword",
                "href",
                "link",
                "uri",
                "cmd",
                "shell",
                "line",
                "operation",
                "git_op",
                "subcommand",
                "sub_cmd",
                "cwd",
                "repo",
                "work_tree",
                "files",
                "file",
                "file_paths",
            }
        )
        for k in list(out.keys()):
            if k in _AGENT_TOP_LEVEL_PARAM_KEYS or k in _extra_top:
                if k != "parameters":
                    del out[k]
    else:
        out["parameters"] = {}

    return out


def parse_agent_json(resp_text, deps: AgentJsonDeps) -> dict:
    """Parse model output into a dict. Handles markdown fences, partial JSON, and plain text."""
    if resp_text is None or (isinstance(resp_text, str) and not resp_text.strip()):
        return {"action": "answer", "answer": "No response from model."}
    text = resp_text.strip()
    # Strip ```json ... ``` fences if present
    fence = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    candidates = []
    if text:
        candidates.append(text)
        tail = clean_json_response(text)
        if tail and tail != text:
            candidates.append(tail)

    for candidate in candidates:
        if not candidate:
            continue
        parsed = try_json_loads_object(candidate)
        if isinstance(parsed, dict):
            return normalize_agent_dict(parsed, deps)

    # Balanced `{...}` spans (avoids broken first-{ to last-} when multiple objects exist)
    best = best_agent_dict_from_text(text, deps.all_known_tools())
    if best:
        return normalize_agent_dict(best, deps)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            start = candidate.index("{")
            end = candidate.rindex("}") + 1
            parsed = try_json_loads_object(candidate[start:end])
            if isinstance(parsed, dict):
                return normalize_agent_dict(parsed, deps)
        except ValueError:
            continue

    # Last resort: treat as a direct answer (general Q&A without valid JSON)
    return normalize_agent_dict({"action": "answer", "answer": text}, deps)
