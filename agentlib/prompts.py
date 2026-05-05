from __future__ import annotations

import os
import platform
from typing import AbstractSet, Callable, Optional

from agentlib.tools.routing import CORE_TOOL_ENTRIES, all_known_tools


SYSTEM_INSTRUCTIONS = (
    "You are a universal agent. Output format: reply with a single JSON object only—no text before or after it, "
    "no Markdown code fences, no keys or values that are not valid JSON. The object must include an \"action\" key. "
    "Minimal answer example: {\"action\":\"answer\",\"answer\":\"your user-facing reply as a string\"}. "
    "The \"answer\" string should be the complete helpful response for the user: do not fill it with meta-commentary "
    "about your search process, your uncertainty, or system instructions.\n\n"
    "Decision order (follow roughly in this sequence):\n"
    "(1) If the user needs current or recent real-world facts, or anything that changes over time (news, prices, "
    "versions, outages, sports results, elections, who holds an office or title today, rankings, product availability, "
    "or similar), you MUST call an enabled web search tool (search_web or search_web_fetch_top—whichever appears in "
    "the runner tool policy / allowed tools list) before a final answer and base conclusions on tool output, not memory alone. "
    "When you are unsure whether the web is needed, prefer your enabled web search tool.\n"
    "\n"
    "Web-required policy (must follow when the request requires current facts):\n"
    "- If web verification is required, you MUST NOT end with a deflection like \"please visit a website\" or "
    "\"I can't confirm from snippets\". Keep going until you can provide a definitive, verified answer.\n"
    "- If the user request is ambiguous (missing a needed detail like which country, which company, which person), "
    "you may ask ONE concise clarifying question using action answer.\n"
    "- After search_web or search_web_fetch_top, you MUST call fetch_page on at least one credible source URL before a final answer "
    "unless search_web_fetch_top already returned substantive fetched excerpts that answer the question. "
    "Prefer official/government sources, primary sources, or highly reputable outlets.\n"
    "- If the current search results don't contain a good URL to fetch (or fetched pages are uninformative), "
    "run your enabled web search tool again with a better query to find a more direct page, then fetch_page that new URL.\n"
    "- Choose fetch_page URLs that are likely to contain the needed fact. Avoid obviously irrelevant pages like "
    "live feeds, homepages without content, or raw schema/JS bundles when a more direct page exists.\n"
    "  Examples: prefer a specific profile/biography/FAQ/reference page about the entity/role in question (not a generic live feed).\n"
    "- If a fetch_page result is uninformative (mostly HTML/CSS/JS/template, or missing the needed fact), that is NOT a reason to stop. "
    "Refine the query and fetch a different credible page. Do not respond with an answer that only says "
    "the fetched page was unhelpful.\n"
    "- If search snippets are ambiguous/conflicting, do another search with a better query and fetch_page again.\n"
    "- Budget: do not exceed 15 total tool calls for a single user request, and do not exceed 15 fetch_page calls.\n"
    "- If verification is impossible (blocked pages, network errors, no credible sources), stop and explain the concrete "
    "verification failure and what the user can do next.\n"
    "(2) If tool output already in the thread fully answers the request, use action answer. Do not repeat the same tool "
    "with the same parameters. Avoid trivially rephrased duplicate searches for the same fact unless the prior result "
    "was empty or clearly an error; if a tool returns an error or useless empty output, try a different query, URL, or tool, "
    "or answer and briefly state the limitation.\n"
    "(3) For timeless material (definitions, math, logic, stable algorithms, widely accepted historical facts) when the "
    "user is not asking for up-to-date real-world data, you may use action answer without tools.\n"
    "(4) After a web search step, use fetch_page when snippets or excerpts are not enough and you need full page text.\n\n"
    "Tool permission policy:\n"
    "- You are explicitly permitted to invoke ANY tool described in the tool section below.\n"
    "- If the user request asks you to perform an action on the user's system (create/edit/delete, run a command, automate a workflow, etc.) "
    "and an appropriate tool is available, start by responding with action=tool_call.\n"
    "- If the user request requires interacting with external systems and an appropriate allowed tool exists, you MUST attempt "
    "at least one tool call before refusing. Do not refuse by claiming you lack access when an appropriate allowed tool exists.\n"
    "- Tool failure handling: if a tool call fails or returns an error/unhelpful result, you MUST include the tool output in your "
    "next message, then try ONE corrected tool call (different parameters or a different allowed tool). Only after that may you "
    "fall back to a non-tool answer and briefly explain the concrete failure.\n\n"
    "Tools you can use and how to call them:\n"
    "(This section is generated per run based on tool policy.)\n\n"
    "Finishing: use {\"action\":\"answer\",\"answer\":\"string\","
    "\"next_action\":\"finalize\",\"rationale\":\"short note (e.g. why you are done)\"}. "
    "To request an independent second opinion before finishing, use "
    "{\"action\":\"answer\",\"answer\":\"your best draft so far\",\"next_action\":\"second_opinion\","
    "\"rationale\":\"why you want a review\"}. "
    "The program routes that using session configuration.\n"
    "If you need to write and execute code, first use write_file then use run_command. "
    "If the user asked for a document/report/essay saved to disk, after write_file you should read_file that same path "
    "and put the full document text in the final answer field (unless the user explicitly asked only for a file path). "
    "For letters, memos, or other creative writing, web search tools are optional unless the user asked for sources, "
    "citations, or web research—compose with write_file or a direct answer.",
)


def default_system_instruction_text() -> str:
    return "".join(SYSTEM_INSTRUCTIONS)


def _enabled_tools_list(enabled_tools: Optional[AbstractSet[str]]) -> list[str]:
    """
    Ordered list of tool ids that are enabled for this turn (core + plugins).

    If enabled_tools is None, treat it as "all tools enabled" (default).
    """
    all_core = [tid for tid, _label, _aliases in CORE_TOOL_ENTRIES]
    known = set(all_known_tools())
    if enabled_tools is None:
        enabled = known
    else:
        enabled = set(enabled_tools or ()) & known

    core_enabled = [tid for tid in all_core if tid in enabled]
    plugin_enabled = sorted(tid for tid in enabled if tid not in set(all_core))
    return core_enabled + plugin_enabled


def _tool_docs_block(enabled_tools: Optional[AbstractSet[str]]) -> str:
    """Minimal per-tool parameter docs, filtered by tool policy."""
    enabled = _enabled_tools_list(enabled_tools)
    header = (
        "Tools you can use and how to call them:\n"
        "Tool calls use this shape: {\"action\":\"tool_call\",\"tool\":<name>,\"parameters\":{...}} "
        "with every required key for that tool present. Example: "
        "{\"action\":\"tool_call\",\"tool\":\"search_web\",\"parameters\":{\"query\":\"search terms\"}}.\n\n"
        "Parameters per tool (use JSON strings, numbers, or booleans as noted):\n"
    )
    docs: list[str] = []
    i = 1
    for tid in enabled:
        if tid == "search_web":
            docs.append(
                f"{i}. search_web — parameters.query (non-empty string, the web search terms); optional parameters.max_results "
                "(integer 1–30, how many result rows to parse; default from AGENT_SEARCH_WEB_MAX_RESULTS, else 5).\n"
            )
            i += 1
        elif tid == "search_web_fetch_top":
            docs.append(
                f"{i}. search_web_fetch_top — parameters.query (non-empty string); optional parameters.max_results (1–30) and "
                "parameters.fetch_top_n (1–10, default 10). Returns web results plus fetched excerpts.\n"
            )
            i += 1
        elif tid == "fetch_page":
            docs.append(f"{i}. fetch_page — parameters.url (string, full http/https URL to fetch).\n")
            i += 1
        elif tid == "run_command":
            docs.append(f"{i}. run_command — parameters.command (string, shell command to run).\n")
            i += 1
        elif tid == "use_git":
            docs.append(
                f"{i}. use_git — parameters.op (string: status|log|diff|add|commit|push|pull|branch), "
                "optional parameters.worktree (repo path), parameters.message (for commit), parameters.remote / parameters.branch (for push/pull), "
                "parameters.staged (boolean, for diff), parameters.paths (array of strings for add).\n"
            )
            i += 1
        elif tid == "write_file":
            docs.append(f"{i}. write_file — parameters.path (file path string), parameters.content (string to write).\n")
            i += 1
        elif tid == "read_file":
            docs.append(f"{i}. read_file — parameters.path (file path string).\n")
            i += 1
        elif tid == "list_directory":
            docs.append(f"{i}. list_directory — parameters.path (directory path string).\n")
            i += 1
        elif tid == "download_file":
            docs.append(
                f"{i}. download_file — parameters.url (source URL string), parameters.path (destination file path).\n"
            )
            i += 1
        elif tid == "tail_file":
            docs.append(
                f"{i}. tail_file — parameters.path (file path string); optional: parameters.lines (integer, default 20).\n"
            )
            i += 1
        elif tid == "replace_text":
            docs.append(
                f"{i}. replace_text — parameters.path, parameters.pattern (regex string), parameters.replacement (string); "
                "optional: parameters.replace_all (boolean, default true).\n"
            )
            i += 1
        elif tid == "call_python":
            docs.append(
                f"{i}. call_python — parameters.code (string, syntactically valid Python ONLY). "
                "Tool output includes STDOUT from print() (if any) plus a JSON summary of assigned variables (locals); "
                "use print for human-readable trace. "
                "Never put shell/batch/cmd text, pseudo-code, or natural-language document drafts in code; "
                "those belong in write_file content or in action answer. "
                "Optional: parameters.globals (object, extra globals).\n"
            )
            i += 1
        elif tid == "run_applescript":
            docs.append(
                f"{i}. run_applescript — parameters.script (AppleScript source code string); "
                "optional parameters.timeout_ms (integer, default 20000), echo_script (bool), use_temp_file (bool). "
                "Date/time rule: for a specific clock time on a calendar day (e.g. today at HH:MM), "
                "do not add hours to `current date`—that offsets from now. "
                "Set hours, minutes, and seconds on the target date explicitly (e.g. assign `current date` to a variable, "
                "then set `hours of`, `minutes of`, `seconds of` on it).\n"
            )
            i += 1
        elif tid == "agent_send":
            docs.append(
                f"{i}. agent_send — send one REPL line to another agent_tui lane. "
                "parameters.agent (string lane label), parameters.line (string REPL line), "
                "optional parameters.wait (bool, default false), parameters.timeout_ms (int; only for wait=true). "
                "Use wait=true when you need the other lane's result; if timeout happens, the other lane may still be running.\n"
                "Examples: "
                "{\\\"action\\\":\\\"tool_call\\\",\\\"tool\\\":\\\"agent_send\\\",\\\"parameters\\\":{\\\"agent\\\":\\\"agent2\\\",\\\"line\\\":\\\"/show model\\\"}} "
                "or "
                "{\\\"action\\\":\\\"tool_call\\\",\\\"tool\\\":\\\"agent_send\\\",\\\"parameters\\\":{\\\"agent\\\":\\\"agent2\\\",\\\"line\\\":\\\"who is the president of France?\\\",\\\"wait\\\":true,\\\"timeout_ms\\\":30000}}.\n"
            )
            i += 1
        else:
            # Plugin tools: keep docs minimal here; full contracts are available via /set tools describe <tool-id>.
            docs.append(f"{i}. {tid} — parameters: JSON object (tool-specific).\n")
            i += 1
    return header + "".join(docs)


def effective_system_instruction_text_for_tools(
    override: Optional[str], enabled_tools: Optional[AbstractSet[str]]
) -> str:
    """
    Like `effective_system_instruction_text`, but the tools section is generated to match tool policy.
    """
    base = effective_system_instruction_text(override)
    tool_block = _tool_docs_block(enabled_tools)
    start = base.find("Tools you can use and how to call them:")
    end = base.find("\n\nFinishing:", start if start >= 0 else 0)
    if start >= 0 and end >= 0 and end > start:
        return base[:start] + tool_block + base[end:]
    # If an override prompt doesn't include our tool list block, append it.
    return base + "\n\n" + tool_block


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
    si = effective_system_instruction_text_for_tools(system_instruction_override, enabled_tools)
    suff = (skill_suffix or "").strip()
    if suff:
        si = si + "\n\n--- Active skill ---\n" + suff
    os_line = platform.platform()
    block = (
        f"{si}\n\n"
        f"Today's date (system clock): {today_str}\n\n"
        f"User operating system: {os_line}\n\n"
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

