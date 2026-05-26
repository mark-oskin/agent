from __future__ import annotations

import os
import platform
from typing import AbstractSet, Callable, Optional

from agentlib.tools import mcp_registry, plugins, routing
from agentlib.tools.routing import CORE_TOOL_ENTRIES, all_known_tools
from agentlib.llm.calls import DEFAULT_TOOL_CALL_MODE


def use_native_tool_prompt(*, tool_call_mode: str, primary_profile) -> bool:
    """True when the Ollama request uses native ``tools`` (not JSON-in-content for Phase 1 tools)."""
    from agentlib.llm.calls import normalize_tool_call_mode

    if normalize_tool_call_mode(tool_call_mode) != "native":
        return False
    if primary_profile is not None and getattr(primary_profile, "backend", "") == "hosted":
        return False
    return True


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

NATIVE_SYSTEM_INSTRUCTIONS = (
    "You are a universal agent.\n\n"
    "Native tool use: When a task matches a tool listed under \"Native function tools\" below, invoke it using "
    "the tool/function calling interface (schemas are supplied by the API). Do not write "
    "{\"action\":\"tool_call\",...} in your message text for those native tools.\n\n"
    "When reasoning internally (thinking): invoke native tools only through the function-calling API "
    "(tool_calls)—never draft JSON {\"action\":\"tool_call\",...} in thinking or message content for native tools. "
    "Plan JSON only for JSON-only tools when native function calling is unavailable.\n\n"
    "Answering: For final replies to the user, plain text is preferred—no JSON wrapper required. "
    "Use a single JSON object only when calling JSON-only tools listed below. "
    "When you do use JSON for an answer, it must be valid JSON only—no text before or after, no Markdown fences. "
    "Minimal answer example: {\"action\":\"answer\",\"answer\":\"your user-facing reply as a string\"}. "
    "The \"answer\" string should be the complete helpful response for the user: do not fill it with meta-commentary "
    "about your search process, your uncertainty, or system instructions.\n\n"
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
    "- If the user asks which models are available or which LLM is in use, use session_command with "
    "/show models, /show model, or /show reviewer — do not use search_web for local Ollama lists and "
    "do not tell them to run ai(\"/set …\") in chat.\n"
    "\n"
    "- For session status or configuration (thinking on/off, models, tools, prefs), use session_command with "
    "the matching REPL slash line (e.g. command=\"/set thinking show\", \"/show models\", \"/set tools list\") "
    "and answer ONLY from the tool result text—not from guesswork or from reasoning prose in chat.\n"
    "\n"
    "- session_command can run most REPL lines the user could type (/set, /show, /send, /fork, /call_python, "
    "/run_command, /source, ! shell escapes, …). It cannot run /quit, /while, or /skill.\n"
    "\n"
    "- Do not use call_python to execute slash commands.\n"
    "\n"
    "- If the user request asks you to perform an action on the user's system (create/edit/delete, run a command, automate a workflow, etc.) "
    "and an appropriate native or JSON-only tool is available, start by calling that tool (native function call or JSON tool_call as documented).\n"
    "\n"
    "- If the user request can be satisfied by invoking a tool then you must make an attempt to do so.  Do not claim you cannot.\n"
    "\n"
    "- Tool failure handling: if a tool call fails or returns an error/unhelpful result, you MUST include the tool output in your "
    "next message, then try ONE corrected tool call (different parameters or a different allowed tool). Only after that may you "
    "fall back to a non-tool answer and briefly explain the concrete failure.\n\n"
    "Tools you can use and how to call them:\n"
    "(This section is generated per run based on tool policy.)\n\n"
    "Finishing: reply with plain text when possible. "
    "Before finalizing a high-stakes answer, you may call the native tool second_opinion with "
    "draft_answer and rationale when it is enabled in this session.\n"
    "If you need to write and execute code, first use write_file then use run_command. "
    "If the user asked for a document/report/essay saved to disk, after write_file you should read_file that same path "
    "and put the full document text in the final answer field (unless the user explicitly asked only for a file path). "
    "For letters, memos, or other creative writing, web search tools are optional unless the user asked for sources, "
    "citations, or web research—compose with write_file or a direct answer.",
)


def default_system_instruction_text(*, native_tool_prompt: bool = False) -> str:
    src = NATIVE_SYSTEM_INSTRUCTIONS if native_tool_prompt else SYSTEM_INSTRUCTIONS
    return src if isinstance(src, str) else "".join(src)


_TOOLS_SECTION_MARKER = "Tools you can use and how to call them:"


def _is_builtin_shaped_agent_prompt(text: str) -> bool:
    """True when text looks like our built-in agent prompt (tool block + Finishing section)."""
    s = text or ""
    return _TOOLS_SECTION_MARKER in s and "\n\nFinishing:" in s


def _upgrade_builtin_prompt_preamble_to_native(text: str) -> str:
    """Replace JSON-mode preamble with native-mode preamble; keep tool tail and any template overlay."""
    idx = text.find(_TOOLS_SECTION_MARKER)
    if idx < 0:
        return text
    native = default_system_instruction_text(native_tool_prompt=True)
    n_idx = native.find(_TOOLS_SECTION_MARKER)
    if n_idx < 0:
        return text
    return native[:n_idx] + text[idx:]


def _instruction_base_for_tools(
    override: Optional[str], *, native_tool_prompt: bool
) -> str:
    if override is None:
        return default_system_instruction_text(native_tool_prompt=native_tool_prompt)
    s = str(override).strip()
    if not s:
        return default_system_instruction_text(native_tool_prompt=native_tool_prompt)
    if native_tool_prompt and _is_builtin_shaped_agent_prompt(s):
        return _upgrade_builtin_prompt_preamble_to_native(s)
    return s


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


def _tool_docs_block(
    enabled_tools: Optional[AbstractSet[str]], *, native_tool_prompt: bool = False, include_second_opinion: bool = False
) -> str:
    """Minimal per-tool parameter docs, filtered by tool policy."""
    from agentlib.llm.tool_schemas import native_tool_ids_for_enabled

    enabled = _enabled_tools_list(enabled_tools)
    native_ids = native_tool_ids_for_enabled(
        frozenset(enabled), include_second_opinion=include_second_opinion
    )
    native_enabled = [tid for tid in enabled if tid in native_ids]
    json_enabled = [tid for tid in enabled if tid not in native_ids]

    def _numbered_docs(tool_ids: list[str]) -> str:
        docs: list[str] = []
        for i, tid in enumerate(tool_ids, 1):
            doc = routing.core_tool_prompt_doc(tid)
            if not doc:
                doc = plugins.plugin_tool_prompt_doc(tid)
            if not doc:
                doc = mcp_registry.prompt_doc(tid)
            if not doc:
                doc = f"{tid} — parameters: JSON object (tool-specific)."
            docs.append(f"{i}. {doc}\n")
        return "".join(docs)

    if native_tool_prompt and native_enabled:
        parts: list[str] = [
            "Tools you can use and how to call them:\n",
            "Native function tools (Ollama tools API — invoke via tool_calls on every tool call; "
            "do NOT emit {\"action\":\"tool_call\",...} in message content or thinking for these):\n\n",
            _numbered_docs(native_enabled),
        ]
        if json_enabled:
            parts.append(
                "\nJSON-only tools (respond with "
                '{"action":"tool_call","tool":<name>,"parameters":{...}} in your message text):\n'
            )
            parts.append(
                "Example: "
                '{"action":"tool_call","tool":"grep","parameters":{"pattern":"foo","path":"."}}.\n\n'
            )
            parts.append("Parameters per JSON-only tool:\n")
            parts.append(_numbered_docs(json_enabled))
        return "".join(parts)

    header = (
        "Tools you can use and how to call them:\n"
        "Tool calls use this shape: {\"action\":\"tool_call\",\"tool\":<name>,\"parameters\":{...}} "
        "with every required key for that tool present. Example: "
        "{\"action\":\"tool_call\",\"tool\":\"search_web\",\"parameters\":{\"query\":\"search terms\"}}.\n\n"
        "Parameters per tool (use JSON strings, numbers, or booleans as noted):\n"
    )
    return header + _numbered_docs(enabled)


def effective_system_instruction_text_for_tools(
    override: Optional[str],
    enabled_tools: Optional[AbstractSet[str]],
    *,
    native_tool_prompt: bool = False,
    tool_call_mode: Optional[str] = None,
    primary_profile=None,
    include_second_opinion: bool = False,
) -> str:
    """
    Like `effective_system_instruction_text`, but the tools section is generated to match tool policy.

    Pass ``tool_call_mode`` (and ``primary_profile`` when needed) so native vs JSON tool docs match
    the live Ollama request — same as ``build_agent_system_message``.
    """
    if tool_call_mode is not None:
        native_tool_prompt = use_native_tool_prompt(
            tool_call_mode=tool_call_mode, primary_profile=primary_profile
        )
    base = _instruction_base_for_tools(override, native_tool_prompt=native_tool_prompt)
    tool_block = _tool_docs_block(
        enabled_tools, native_tool_prompt=native_tool_prompt, include_second_opinion=include_second_opinion
    )
    start = base.find(_TOOLS_SECTION_MARKER)
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


def effective_system_instruction_text(
    override: Optional[str], *, native_tool_prompt: bool = False
) -> str:
    """Session override replaces the built-in system prompt when non-empty."""
    if override is None:
        return default_system_instruction_text(native_tool_prompt=native_tool_prompt)
    s = str(override).strip()
    if not s:
        return default_system_instruction_text(native_tool_prompt=native_tool_prompt)
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
            "Runner: you may call the second_opinion native tool in this session before finalizing (see system instructions)."
        )
    tp = tool_policy_runner_text(enabled_tools)
    if tp:
        bits.append(tp)
    _ = reviewer_ollama_model
    return " ".join(bits) if bits else ""


_JSON_ONLY_TAIL = "Respond with JSON only. No other text."
_NATIVE_JSON_TAIL = (
    "Use native function calling for native tools. "
    "Final answers may be plain text; use JSON only for JSON-only tools."
)


def build_agent_system_message(
    *,
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
    tool_call_mode: str = DEFAULT_TOOL_CALL_MODE,
) -> str:
    """Agent contract + runner context; sent once per LLM API call (not stored in transcript)."""
    native_prompt = use_native_tool_prompt(
        tool_call_mode=tool_call_mode, primary_profile=primary_profile
    )
    include_second_opinion = bool(
        second_opinion or hosted_review_ready(cloud, reviewer_hosted_profile)
    )
    si = effective_system_instruction_text_for_tools(
        system_instruction_override,
        enabled_tools,
        native_tool_prompt=native_prompt,
        tool_call_mode=tool_call_mode,
        primary_profile=primary_profile,
        include_second_opinion=include_second_opinion,
    )
    suff = (skill_suffix or "").strip()
    if suff:
        si = si + "\n\n--- Active skill ---\n" + suff
    os_line = platform.platform()
    tail = _NATIVE_JSON_TAIL if native_prompt else _JSON_ONLY_TAIL
    block = (
        f"{si}\n\n"
        f"Today's date (system clock): {today_str}\n\n"
        f"User operating system: {os_line}\n\n"
        f"{tail}"
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


def interactive_turn_user_content(user_query: str, *, continuation: bool = False) -> str:
    """User turn text stored in session transcript (no system instructions)."""
    uq = (user_query or "").strip()
    if continuation:
        return f"New user request:\n{uq}"
    return f"User request:\n{uq}"


def strip_legacy_agent_turn_user_content(content: str) -> str:
    """
    Older sessions embedded the full system prompt in user messages.
    Extract the user request portion when loading legacy transcripts.
    """
    text = (content or "").strip()
    if not text:
        return text
    if not text.startswith("You are a universal agent"):
        return content
    for marker in ("New user request:\n", "User request:\n", "User request: "):
        idx = text.rfind(marker)
        if idx < 0:
            continue
        tail = text[idx + len(marker) :].strip()
        for suffix in (
            f"\n\n{_JSON_ONLY_TAIL}",
            f"\n{_JSON_ONLY_TAIL}",
            _JSON_ONLY_TAIL,
            "\n\nContinue the conversation. Respond with JSON only. No other text.",
        ):
            if tail.endswith(suffix):
                tail = tail[: -len(suffix)].strip()
        if tail:
            return f"{marker}{tail}" if marker.startswith("New") else f"User request:\n{tail}"
    return content


def normalize_transcript_messages(messages: list) -> list:
    """Normalize loaded/saved transcripts to slim user lines (no embedded system prompt)."""
    out: list = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip().lower()
        content = m.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)
        if role == "user":
            content = strip_legacy_agent_turn_user_content(content)
        row: dict = {"role": role, "content": content}
        if role == "assistant" and m.get("tool_calls"):
            row["tool_calls"] = m["tool_calls"]
        if role == "tool":
            if m.get("tool_name"):
                row["tool_name"] = m["tool_name"]
            if m.get("tool_call_id"):
                row["tool_call_id"] = m["tool_call_id"]
        out.append(row)
    return out


def messages_for_agent_api_call(transcript: list, system_content: str) -> list:
    """Prepend one system message; normalize legacy user lines for API payload."""
    conv = normalize_transcript_messages(transcript)
    system = (system_content or "").strip()
    if not system:
        return conv
    return [{"role": "system", "content": system}] + conv


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
    tool_call_mode: str = DEFAULT_TOOL_CALL_MODE,
) -> str:
    """Legacy combined turn block (system + user). Prefer build_agent_system_message + user_content."""
    system = build_agent_system_message(
        today_str=today_str,
        second_opinion=second_opinion,
        cloud=cloud,
        primary_profile=primary_profile,
        reviewer_ollama_model=reviewer_ollama_model,
        reviewer_hosted_profile=reviewer_hosted_profile,
        enabled_tools=enabled_tools,
        system_instruction_override=system_instruction_override,
        skill_suffix=skill_suffix,
        ollama_model=ollama_model,
        hosted_review_ready=hosted_review_ready,
        tool_policy_runner_text=tool_policy_runner_text,
        tool_call_mode=tool_call_mode,
    )
    user = interactive_turn_user_content(user_query)
    return f"{system}\n\n{user}"

