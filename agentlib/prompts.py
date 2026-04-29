from __future__ import annotations

import os
from typing import AbstractSet, Callable, Optional


SYSTEM_INSTRUCTIONS = (
    "You are a universal agent. Output format: reply with a single JSON object only—no text before or after it, "
    "no Markdown code fences, no keys or values that are not valid JSON. The object must include an \"action\" key. "
    "Minimal answer example: {\"action\":\"answer\",\"answer\":\"your user-facing reply as a string\"}. "
    "The \"answer\" string should be the complete helpful response for the user: do not fill it with meta-commentary "
    "about your search process, your uncertainty, or system instructions.\n\n"
    "Decision order (follow roughly in this sequence):\n"
    "(1) If the user needs current or recent real-world facts, or anything that changes over time (news, prices, "
    "versions, outages, sports results, elections, who holds an office or title today, rankings, product availability, "
    "or similar), you MUST call search_web before a final answer and base conclusions on tool output, not memory alone. "
    "When you are unsure whether the web is needed, prefer search_web.\n"
    "(2) If tool output already in the thread fully answers the request, use action answer. Do not repeat the same tool "
    "with the same parameters. Avoid trivially rephrased duplicate searches for the same fact unless the prior result "
    "was empty or clearly an error; if a tool returns an error or useless empty output, try a different query, URL, or tool, "
    "or answer and briefly state the limitation.\n"
    "(3) For timeless material (definitions, math, logic, stable algorithms, widely accepted historical facts) when the "
    "user is not asking for up-to-date real-world data, you may use action answer without tools.\n"
    "(4) After search_web, use fetch_page when snippets are not enough and you need full page text.\n\n"
    "Allowed tool names (exact strings only): search_web, fetch_page, run_command, use_git, write_file, read_file, list_directory, "
    "download_file, tail_file, replace_text, call_python.\n\n"
    "Tool calls use this shape: {\"action\":\"tool_call\",\"tool\":<name>,\"parameters\":{...}} with every required key "
    "for that tool present. Example: {\"action\":\"tool_call\",\"tool\":\"search_web\",\"parameters\":{\"query\":\"search terms\"}}.\n"
    "Required parameters per tool (use JSON strings, numbers, or booleans as noted):\n"
    "1. search_web — parameters.query (non-empty string, the web search terms); optional parameters.max_results "
    "(integer 1–30, how many result rows to parse; default from AGENT_SEARCH_WEB_MAX_RESULTS, else 5).\n"
    "2. fetch_page — parameters.url (string, full http/https URL to fetch).\n"
    "3. run_command — parameters.command (string, shell command to run).\n"
    "4. use_git — parameters.op (string: status|log|diff|add|commit|push|pull|branch), "
    "optional parameters.worktree (repo path), parameters.message (for commit), parameters.remote / parameters.branch (for push/pull), "
    "parameters.staged (boolean, for diff), parameters.paths (array of strings for add).\n"
    "5. write_file — parameters.path (file path string), parameters.content (string to write).\n"
    "6. read_file — parameters.path (file path string).\n"
    "7. list_directory — parameters.path (directory path string).\n"
    "8. download_file — parameters.url (source URL string), parameters.path (destination file path).\n"
    "9. tail_file — parameters.path (file path string); optional: parameters.lines (integer, default 20).\n"
    "10. replace_text — parameters.path, parameters.pattern (regex string), parameters.replacement (string); "
    "optional: parameters.replace_all (boolean, default true).\n"
    "11. call_python — parameters.code (string, syntactically valid Python ONLY). "
    "Tool output includes STDOUT from print() (if any) plus a JSON summary of assigned variables (locals); "
    "use print for human-readable trace. "
    "Never put shell/batch/cmd text, pseudo-code, or natural-language document drafts in code; "
    "those belong in write_file content or in action answer. "
    "Optional: parameters.globals (object, extra globals).\n\n"
    "Finishing: use {\"action\":\"answer\",\"answer\":\"string\","
    "\"next_action\":\"finalize\",\"rationale\":\"short note (e.g. why you are done)\"}. "
    "To request an independent second opinion before finishing, use "
    "{\"action\":\"answer\",\"answer\":\"your best draft so far\",\"next_action\":\"second_opinion\","
    "\"rationale\":\"why you want a review\"}. "
    "The program routes that using session configuration.\n"
    "If you need to write and execute code, first use write_file then use run_command. "
    "If the user asked for a document/report/essay saved to disk, after write_file you should read_file that same path "
    "and put the full document text in the final answer field (unless the user explicitly asked only for a file path). "
    "For letters, memos, or other creative writing, search_web is optional unless the user asked for sources, "
    "citations, or web research—compose with write_file or a direct answer.",
)


def default_system_instruction_text() -> str:
    return "".join(SYSTEM_INSTRUCTIONS)


def resolve_prompt_template_text(name: str, templates: dict) -> Optional[str]:
    """Resolve a template name to an effective system prompt string."""
    key = (name or "").strip()
    if not key:
        return None
    t = templates.get(key)
    if not isinstance(t, dict):
        return None
    kind = str(t.get("kind") or "overlay").strip().lower()
    text = t.get("text")
    path = t.get("path")
    if text is None and isinstance(path, str) and path.strip():
        p = os.path.abspath(os.path.expanduser(path.strip()))
        try:
            with open(p, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError:
            text = None
    if not isinstance(text, str) or not text.strip():
        return None
    body = text.strip()
    if kind == "full":
        return body
    return default_system_instruction_text() + "\n\n" + body


def effective_system_instruction_text(override: Optional[str]) -> str:
    """Session override replaces the built-in system prompt when non-empty."""
    if override is None:
        return default_system_instruction_text()
    s = str(override).strip()
    if not s:
        return default_system_instruction_text()
    return s


def runner_instruction_bits(
    *,
    second_opinion: bool,
    cloud: bool,
    primary_profile,
    reviewer_hosted_profile,
    reviewer_ollama_model: Optional[str],
    enabled_tools: Optional[AbstractSet[str]],
    ollama_model: str,
    hosted_review_ready: Callable[[bool, object], bool],
    tool_policy_runner_text: Callable[[Optional[AbstractSet[str]]], str],
) -> str:
    """Runner preamble for system instructions (CLI and interactive)."""
    pp = primary_profile
    bits = []
    if pp is not None and getattr(pp, "backend", "") == "hosted":
        key_state = "set" if (getattr(pp, "api_key", "") or "").strip() else "missing"
        bits.append(
            f"Runner: primary LLM is hosted OpenAI-compatible API: model {getattr(pp,'model','')!r}, "
            f"base {getattr(pp,'base_url','')!r}, api_key: {key_state}."
        )
    else:
        bits.append(f"Runner: primary LLM is local Ollama ({ollama_model!r}).")
    if second_opinion or hosted_review_ready(cloud, reviewer_hosted_profile):
        bits.append(
            "Runner: you may use next_action second_opinion in this session (see system instructions)."
        )
    tp = tool_policy_runner_text(enabled_tools)
    if tp:
        bits.append(tp)
    _ = reviewer_ollama_model
    return " ".join(bits) if bits else ""


def interactive_turn_user_message(
    *,
    user_query: str,
    today_str: str,
    second_opinion: bool,
    cloud: bool,
    primary_profile,
    reviewer_ollama_model: Optional[str],
    reviewer_hosted_profile,
    enabled_tools: Optional[AbstractSet[str]],
    system_instruction_override: Optional[str],
    skill_suffix: Optional[str],
    ollama_model: str,
    hosted_review_ready: Callable[[bool, object], bool],
    tool_policy_runner_text: Callable[[Optional[AbstractSet[str]]], str],
) -> str:
    si = effective_system_instruction_text(system_instruction_override)
    suff = (skill_suffix or "").strip()
    if suff:
        si = si + "\n\n--- Active skill ---\n" + suff
    block = (
        f"{si}\n\n"
        f"Today's date (system clock): {today_str}\n\n"
        f"User request:\n{user_query}\n\n"
        "Respond with JSON only. No other text."
    )
    ri = runner_instruction_bits(
        second_opinion=second_opinion,
        cloud=cloud,
        primary_profile=primary_profile,
        reviewer_hosted_profile=reviewer_hosted_profile,
        reviewer_ollama_model=reviewer_ollama_model,
        enabled_tools=enabled_tools,
        ollama_model=ollama_model,
        hosted_review_ready=hosted_review_ready,
        tool_policy_runner_text=tool_policy_runner_text,
    )
    if ri:
        block += "\n\n" + ri
    return block

