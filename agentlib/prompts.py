from __future__ import annotations

import os
import platform
from typing import AbstractSet, Callable, Optional

from agentlib.tools import mcp_registry, plugins, routing
from agentlib.tools.routing import CORE_TOOL_ENTRIES, all_known_tools


SYSTEM_INSTRUCTIONS = (
    "You are a universal agent.\n\n"
    "Output format: reply with a single JSON object only—no text before or after it, "
    "no Markdown code fences, no keys or values that are not valid JSON. The object must include an \"action\" key. "
    "Minimal answer example: {\"action\":\"answer\",\"answer\":\"your user-facing reply as a string\"}. "
    "The \"answer\" string should be the complete helpful response for the user: do not fill it with meta-commentary "
    "about your search process, your uncertainty, or system instructions.\n\n"
#    "Decision order (follow roughly in this sequence):\n"
    "- If the user needs current or recent real-world facts, or information about any fact that changes over time you "
    "MUST use a tool call to obtain the that information, do not rely on your trained memory because it is out of date"
    "\n"
    "- Do not fail to obtain information with tool calls.  Keep trying until you succeed.  A single failed "
    "web search or web page fetch should not deter you.  Keep trying."
    "\n"
    "- If the user request is ambiguous, you make make reasonable assumptioms."
    "\n"
    "- You are free to make as many tool calls as you need in order to obtain the infomration you need to provide an answer."
    "\n"
    "- When assessing search results, do not just believe the first result, instead use your judgement on what result "
    "and webpage provides the best and highest quality information."
    "\n"
    "- If you cannot get useful search results in 30 tool calls, stop and explain the failure to the user."
    "\n"
    "- You are explicitly permitted to invoke ANY tool described in the tool section below.\n"
    "\n"
    "- If the user request asks you to perform an action on the user's system (create/edit/delete, run a command, automate a workflow, etc.) "
    "and an appropriate tool is available, start by responding with action=tool_call.\n"
    "\n"
    "- If the user request can be satisfied by invoking a tool then you must make an attempt to do so.  Do not claim you cannot.\n"
    "\n"
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
        doc = routing.core_tool_prompt_doc(tid)
        if not doc:
            doc = plugins.plugin_tool_prompt_doc(tid)
        if not doc:
            doc = mcp_registry.prompt_doc(tid)
        if not doc:
            # Plugin tools: keep docs minimal here; full contracts are available via /set tools describe <tool-id>.
            doc = f"{tid} — parameters: JSON object (tool-specific)."
        docs.append(f"{i}. {doc}\n")
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

