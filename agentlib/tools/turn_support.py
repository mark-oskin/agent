"""Tool-parameter normalization, dedupe fingerprints, recovery, and tool-result UX."""

from __future__ import annotations

import json
import re
from typing import AbstractSet, Callable, Optional, Tuple

from agentlib.agent_json import try_json_loads_object
from agentlib.coercion import scalar_to_str


TOOL_RECOVERY_TOOLS = frozenset({"run_command", "call_python", "search_web", "fetch_page"})


def merge_tool_param_aliases(tool: str, params: dict, *, scalar_to_str_fn=scalar_to_str) -> dict:
    """Map alternate parameter names models use into the keys our tools expect."""
    p = dict(params) if isinstance(params, dict) else {}
    st = scalar_to_str_fn
    if tool == "search_web":
        if not st(p.get("query"), "").strip():
            for alt in ("q", "search", "keywords", "keyword", "text"):
                t = st(p.get(alt), "").strip()
                if t:
                    p["query"] = t
                    p.pop(alt, None)
                    break
    elif tool == "fetch_page":
        if not st(p.get("url"), "").strip():
            for alt in ("href", "link", "uri"):
                t = st(p.get(alt), "").strip()
                if t:
                    p["url"] = t
                    p.pop(alt, None)
                    break
    elif tool == "run_command":
        if not st(p.get("command"), "").strip():
            for alt in ("cmd", "shell", "line"):
                t = st(p.get(alt), "").strip()
                if t:
                    p["command"] = t
                    p.pop(alt, None)
                    break
    elif tool == "use_git":
        if not st(p.get("op"), "").strip():
            for alt in ("operation", "git_op", "subcommand", "sub_cmd"):
                t = st(p.get(alt), "").strip()
                if t:
                    p["op"] = t
                    p.pop(alt, None)
                    break
        if not st(p.get("worktree"), "").strip():
            for alt in ("cwd", "repo", "work_tree", "path_root"):
                t = st(p.get(alt), "").strip()
                if t:
                    p["worktree"] = t
                    p.pop(alt, None)
                    break
        if not st(p.get("message"), "").strip():
            for alt in ("m", "msg", "commit_message"):
                t = st(p.get(alt), "").strip()
                if t:
                    p["message"] = t
                    p.pop(alt, None)
                    break
        if p.get("paths") is None:
            for alt in ("files", "file", "file_paths"):
                if p.get(alt) is not None:
                    p["paths"] = p.get(alt)
                    p.pop(alt, None)
                    break
    elif tool == "write_file":
        if not st(p.get("content"), "").strip():
            for alt in ("body", "text", "contents", "data"):
                t = st(p.get(alt), "").strip()
                if t:
                    p["content"] = t
                    p.pop(alt, None)
                    break
    return p


def ensure_tool_defaults(tool: str, params: dict, user_query: str, *, scalar_to_str_fn=scalar_to_str) -> dict:
    """Fill required parameters when the model emits an empty object."""
    p = dict(params) if isinstance(params, dict) else {}
    st = scalar_to_str_fn
    uq = (user_query or "").strip()
    if tool == "search_web":
        if not st(p.get("query"), "").strip():
            p["query"] = uq if uq else "web search"
    return p


def tool_params_fingerprint(
    tool: str,
    params,
    *,
    scalar_to_str_fn=scalar_to_str,
    search_web_effective_max_results: Callable[[object], int],
) -> str:
    """Stable key for deduplicating identical tool calls."""
    if not isinstance(params, dict):
        params = {}
    if tool == "search_web":
        q = scalar_to_str_fn(params.get("query"), "").strip()
        qn = re.sub(r"\s+", " ", q).lower().strip(" \t.?!")
        mrx = search_web_effective_max_results(params)
        return f"{tool}\0{json.dumps({'query': qn, 'max_results': mrx}, sort_keys=True, ensure_ascii=False)}"
    return f"{tool}\0{json.dumps(params, sort_keys=True, ensure_ascii=False)}"


def web_tool_result_followup_hint(tool: str, result: str) -> str:
    """
    Short, tool-specific nudge when web tools return errors or no usable data,
    so the model can recover without an extra model round in some clients.
    """
    r = (result or "").strip()
    if not r:
        return ""
    t = (tool or "").strip().lower()
    if t == "fetch_page":
        if "Fetch error:" in r[:400] or r.startswith("Fetch error:"):
            return (
                "The page fetch did not return usable content. If HTTP 4xx/5xx: try another "
                "URL (official docs, archive, or a different path). Use search_web to find a "
                "credible page if you do not have a working link. Do not use run_command with curl/wget."
            )
    if t == "search_web":
        if "No results found" in r:
            return (
                "Narrow or rephrase the query, add a product/site/organization name, or include a year; "
                "or fetch a URL the user gave with fetch_page."
            )
        if "anomaly-modal" in r or "bot-check" in r or "bots use DuckDuckGo" in r:
            return (
                "DuckDuckGo may be rate-limiting. Rely on Wikipedia/inline results if present, "
                "or use fetch_page on a URL from the user or a known-credible source."
            )
    return ""


def is_tool_result_weak_for_dedup(result: str) -> bool:
    """If true, search_web output does not count toward saw_strong_web_result (no URL-backed sources)."""
    r = (result or "").strip()
    if not r:
        return True
    if "No results found" in r:
        return True
    if r.startswith("Fetch error:") or "Fetch error:" in r[:200]:
        return True
    if r.startswith("Read error:") or r.startswith("List dir error:"):
        return True
    if ("[DuckDuckGo instant answer]" in r or "[Web results]" in r or "[Wikipedia search]" in r) and not re.search(
        r"https?://", r
    ):
        # DuckDuckGo HTML links are often scheme-relative redirects:
        #   //duckduckgo.com/l/?uddg=https%3A%2F%2F...
        # Treat encoded destination URLs as URL-backed for verification gating.
        low = r.lower()
        if "uddg=" in low and "duckduckgo" in low:
            return False
        return True
    return False


def tool_result_user_message(
    tool: str,
    params: dict,
    result: str,
    *,
    deliverable_reminder: str = "",
    tool_output_max: int,
    scalar_to_str_fn=scalar_to_str,
) -> str:
    """User follow-up after a tool run so the model reads output and stops re-querying."""
    params_s = json.dumps(params, ensure_ascii=False) if params else "{}"
    max_len = max(1, int(tool_output_max))
    body = result if isinstance(result, str) else str(result)
    if len(body) > max_len:
        body = body[:max_len] + "\n...[truncated for length; use what is shown above]"
    extra = f"\n{deliverable_reminder}\n" if deliverable_reminder else ""
    wfh = web_tool_result_followup_hint((tool or ""), body)
    wfn = f"\n--- Web tool follow-up hint ---\n{wfh}\n" if wfh else ""
    return (
        f"Tool `{tool}` finished.\n"
        f"Parameters: {params_s}\n\n"
        f"Output:\n{body}\n\n"
        f"{extra}{wfn}"
        "Using the output above (and earlier steps in this chat if any), respond with JSON only. "
        "IMPORTANT: Treat tool output as authoritative. If the tool output conflicts with your prior knowledge "
        "or training data, trust the tool output. "
        "If the tool output does not contain any concrete sources/URLs, treat it as insufficient for factual claims "
        "that could be outdated, and call search_web again with a better query or call fetch_page on a credible URL. "
        "If snippets are not enough to verify or resolve ambiguity, use fetch_page on a credible URL "
        "(do NOT use run_command with curl/wget to scrape web pages). "
        "If the user’s question is now answered, respond with "
        '{"action":"answer","answer":"..."} '
        "and nothing else. "
        "Only use {\"action\":\"tool_call\",...} if the output above is empty, is clearly an error, "
        "or is still missing facts you need (or contains conflicting/ambiguous facts that you must resolve) — "
        "and do not repeat the same tool with the same parameters as a previous step unless that step failed."
    )


def tool_result_indicates_retryable_failure(tool: str, result) -> bool:
    """True when output looks like a hard tool failure (not normal STDERR on stderr)."""
    r = (result if isinstance(result, str) else str(result)).strip()
    if not r:
        return False
    if tool == "run_command":
        return r.startswith("Command error:")
    if tool == "call_python":
        return r.startswith("Exec error:")
    if tool == "fetch_page":
        return r.startswith("Fetch error:") or "Fetch error:" in r[:200]
    if tool == "search_web":
        return "No results found" in r
    return False


def parse_tool_recovery_payload(resp_text: str) -> Optional[Tuple[dict, str]]:
    """Parse a recovery-only JSON object (not normalized as agent tool_call JSON)."""
    if not (resp_text or "").strip():
        return None
    text = resp_text.strip()
    fence = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    parsed = try_json_loads_object(text)
    if not isinstance(parsed, dict):
        return None
    if (parsed.get("recovery") or "").strip().lower() != "retry":
        return None
    p = parsed.get("parameters")
    if not isinstance(p, dict):
        return None
    rationale = scalar_to_str(parsed.get("rationale"), "").strip()
    return p, rationale


def suggest_tool_recovery_params(
    tool: str,
    params: dict,
    result: str,
    user_query: str,
    primary_profile,
    et: AbstractSet[str],
    verbose: int,
    *,
    call_ollama_chat: Callable[..., str],
    merge_aliases: Callable[[str, dict], dict],
    ensure_defaults: Callable[[str, dict, str], dict],
) -> Optional[Tuple[dict, str]]:
    """Ask the primary model for corrected parameters for one retry."""
    ctx = {
        "tool": tool,
        "parameters": params,
        "error_output": (result if isinstance(result, str) else str(result))[:8000],
        "user_request": (user_query or "")[:4000],
    }
    rec_extra = (
        "For run_command, parameters.command must be the full shell command string. "
        "For call_python, parameters.code must be syntactically valid Python (string). "
        "For search_web, parameters.query must be a different, non-empty search string than before "
        "(e.g. shorter, alternate keywords, add a year, or a site/product name). "
        "For fetch_page, parameters.url must be a different full http(s) URL (not the same as before): "
        "e.g. official site, different path, or a URL from search results."
    )
    prompt = (
        "A tool run failed or returned no usable data inside an autonomous agent. Read the error, "
        "then propose fixed parameters for exactly ONE retry of the SAME tool (same tool name).\n\n"
        f"Context JSON:\n{json.dumps(ctx, ensure_ascii=False, indent=2)}\n\n"
        "Respond with JSON only, a single object, no markdown:\n"
        '{"recovery":"retry","parameters":{...},"rationale":"one short line"}\n'
        'or {"recovery":"abort","rationale":"why a blind retry is unsafe or useless"}.\n'
        f"{rec_extra}"
    )
    raw = call_ollama_chat(
        [{"role": "user", "content": prompt}],
        primary_profile,
        et,
        verbose=verbose,
    )
    parsed = parse_tool_recovery_payload(raw)
    if not parsed:
        if verbose >= 1:
            print("[*] Tool recovery: no retry proposal (recovery≠retry or invalid JSON).")
        return None
    new_params, rationale = parsed
    new_params = merge_aliases(tool, new_params)
    new_params = ensure_defaults(tool, new_params, user_query)
    return new_params, rationale or "(no rationale)"


def confirm_tool_recovery_retry(
    tool: str,
    old_params: dict,
    new_params: dict,
    rationale: str,
    *,
    interactive_tool_recovery: bool,
    stdin_isatty: bool,
) -> bool:
    """Log model-proposed recovery; always proceed with one automatic retry (no y/N prompt)."""
    if interactive_tool_recovery and stdin_isatty:
        print("\n--- Tool failed; model proposed a corrected retry ---")
        print(f"Tool: {tool}")
        print(f"Rationale: {rationale}")
        print(f"Was: {json.dumps(old_params, ensure_ascii=False)}")
        print(f"Now: {json.dumps(new_params, ensure_ascii=False)}")
    return True
