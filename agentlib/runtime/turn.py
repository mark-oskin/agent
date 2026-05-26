"""Multi-step agent conversation loop (LLM + tools + second opinion)."""

from __future__ import annotations

import re
from typing import AbstractSet, Any, Optional, Tuple

from agentlib.agent_json import consume_last_json_parse_note
from agentlib.prompts import messages_for_agent_api_call
from agentlib.sink import sink_emit
from agentlib.tools import mcp_registry, turn_support
from agentlib.tools.routing import preferred_web_search_tool

from agentlib.llm.tool_schemas import (
    SECOND_OPINION_TOOL_ID,
    deliverable_full_document_user_content,
    invalid_agent_response_user_content,
    missing_answer_field_user_content,
    second_opinion_limit_user_content,
    second_opinion_unavailable_user_content,
    tool_call_only_nudge,
    tool_transport_uses_native,
    truncated_json_recovery_user_content,
    web_search_required_user_content,
)
from agentlib.llm.calls import consume_last_ollama_raw_message
from agentlib.llm.second_opinion import run_second_opinion_tool
from agentlib.transcript_shape import (
    assistant_transcript_message,
    first_tool_call_id,
    tool_transcript_message,
)

from .deps import ConversationTurnDeps

_NEEDS_MORE_WEB_RE = re.compile(
    r"(?is)\b("
    r"need to (?:do|perform|run|try) (?:a|another|more|a more specific) (?:web )?search"
    r"|need to search (?:again|more)"
    r"|need to (?:perform|do|run|try) another query"
    r"|i (?:should|must|need to) (?:search|look) (?:it|this) up"
    r"|i (?:can't|cannot) (?:tell|determine|confirm) from (?:these|the) results"
    r")\b"
)


def _answer_requests_more_web(answer_text: str) -> bool:
    """Heuristic: model 'answered' but is really requesting another web lookup."""
    a = (answer_text or "").strip()
    if not a:
        return False
    return bool(_NEEDS_MORE_WEB_RE.search(a))


_DEFLECTS_UNDER_WEB_REQUIRED_RE = re.compile(
    r"(?is)\b("
    r"unable to provide"
    r"|cannot provide"
    r"|can't provide"
    r"|unable to (?:confirm|determine)"
    r"|cannot (?:confirm|determine)"
    r"|can't (?:confirm|determine)"
    r"|do not contain a definitive"
    r"|does not contain"
    r"|doesn't contain"
    r"|missing the current"
    r"|missing the name"
    r"|primarily code"
    r"|schema markup"
    r"|css/structure"
    r"|css styling"
    r"|styling information"
    r"|no factual (?:text|information)"
    r"|contains no factual"
    r"|does not contain any factual"
    r"|fetched content.*(?:does not contain|contains no)"
    r"|html/schema"
    r"|not (?:a )?definitive"
    r"|conflicting"
    r"|ambiguous"
    r"|based solely on (?:the )?provided search snippets"
    r"|please visit"
    r"|recommend checking"
    r"|recommend (?:you )?(?:checking|visiting)"
    r"|i recommend (?:checking|visiting)"
    r"|visit (?:the )?(?:official|primary) source"
    r"|for (?:the )?most accurate.*visit"
    r"|major news"
    r"|reputable news"
    r"|please provide an authoritative link"
    r"|provide an authoritative link"
    r"|i will be unable to definitively answer"
    r"|unable to definitively answer"
    r")\b"
)


def _answer_deflects_instead_of_verifying(answer_text: str) -> bool:
    """Heuristic: model punts user to websites instead of fetching/verifying."""
    a = (answer_text or "").strip()
    if not a:
        return False
    return bool(_DEFLECTS_UNDER_WEB_REQUIRED_RE.search(a))


_CLARIFYING_QUESTION_RE = re.compile(
    r"(?is)\b("
    r"which"
    r"|what"
    r"|do you mean"
    r"|could you"
    r"|can you"
    r"|please clarify"
    r")\b"
)


def _answer_is_clarifying_question(answer_text: str) -> bool:
    """Allow answering with a clarifying question even under web_required."""
    a = (answer_text or "").strip()
    if not a:
        return False
    if a.endswith("?"):
        return True
    return bool(_CLARIFYING_QUESTION_RE.search(a))

_REFUSES_DUE_TO_ACCESS_RE = re.compile(
    r"(?is)\b("
    r"i (?:can't|cannot|do not|don't) (?:directly )?(?:access|interact with|control|modify)"
    r"|i (?:don't|do not) have (?:direct|programmatic) access"
    r"|i am (?:unable|not able) to (?:access|interact|control|modify)"
    r"|no tool is provided"
    r"|no tool (?:is|was) available"
    r"|not (?:possible|able) to (?:access|interact|control|modify)"
    r")\b"
)


def _answer_refuses_despite_tools(answer_text: str) -> bool:
    """Heuristic: model refuses with 'no access' language even though tools may exist."""
    a = (answer_text or "").strip()
    if not a:
        return False
    return bool(_REFUSES_DUE_TO_ACCESS_RE.search(a))


def _transcript_append_assistant_user(
    messages: list,
    raw_msg: Optional[dict],
    response_text: str,
    user_content: str,
    *,
    native_transport: bool,
) -> None:
    messages.append(
        assistant_transcript_message(
            raw_msg, fallback_content=response_text, use_provider_shape=native_transport
        )
    )
    messages.append({"role": "user", "content": user_content})


def _transcript_append_tool_followup(
    messages: list,
    raw_msg: Optional[dict],
    response_text: str,
    tool: str,
    result: str,
    user_followup: str,
    *,
    native_transport: bool,
) -> None:
    asst = assistant_transcript_message(
        raw_msg, fallback_content=response_text, use_provider_shape=native_transport
    )
    body = result if isinstance(result, str) else str(result)
    if native_transport and isinstance(raw_msg, dict) and raw_msg.get("tool_calls"):
        tc_id = first_tool_call_id(raw_msg.get("tool_calls"))
        messages.append(asst)
        messages.append(tool_transcript_message(tool, body, tool_call_id=tc_id))
        if user_followup:
            messages.append({"role": "user", "content": user_followup})
    else:
        messages.append({"role": "assistant", "content": response_text})
        if user_followup:
            messages.append({"role": "user", "content": user_followup})


def run_agent_conversation_turn(
    messages: list,
    user_query: str,
    today_str: str,
    deps: ConversationTurnDeps,
    *,
    web_required: bool,
    deliverable_wanted: bool,
    verbose: int,
    second_opinion_enabled: bool,
    cloud_ai_enabled: bool,
    primary_profile: Any = None,
    reviewer_hosted_profile: Any = None,
    reviewer_ollama_model: Optional[str] = None,
    enabled_tools: Optional[AbstractSet[str]] = None,
    interactive_tool_recovery: bool = False,
    context_cfg: Optional[dict] = None,
    print_answer: bool = True,
    max_agent_steps: int = 30,
    max_agent_steps_web: int = 15,
    max_tool_calls_web: int = 15,
    max_fetch_page_web: int = 15,
    agent_system_message: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    from agentlib.llm import streaming as llm_streaming
    from agentlib.sink import set_sink_show_draft

    set_sink_show_draft(bool(deps.show_draft_enabled()))
    llm_streaming.reset_assistant_answer_streamed()
    et = deps.coerce_enabled_tools(enabled_tools)
    mandatory_web_tool = preferred_web_search_tool(et)
    if web_required and mandatory_web_tool is None:
        web_required = False
    seen_tool_fingerprints: set = set()
    reviewed_tool_need = False
    saw_strong_web_result = False
    answered = False
    tool_executed = False
    tool_calls_executed = 0
    fetch_pages_executed = 0
    second_opinion_rounds = 0
    final_answer: Optional[str] = None
    deliverable_path: Optional[str] = None
    deliverable_read_ok = False
    deliverable_file_chars = 0
    known = deps.all_known_tools()
    ms = max(1, int(max_agent_steps))
    msw = max(1, int(max_agent_steps_web))
    mtcw = max(1, int(max_tool_calls_web))
    fetch_limit = max(1, int(max_fetch_page_web))
    step_limit = msw if web_required else ms
    verified_by_fetch = False
    forced_tool_attempt_after_refusal = False
    tool_call_mode = deps.ollama_tool_call_mode()
    include_second_opinion = bool(
        second_opinion_enabled
        or deps.hosted_review_ready(cloud_ai_enabled, reviewer_hosted_profile)
    )
    for _ in range(step_limit):
        messages = deps.maybe_compact_context_window(
            messages,
            user_query=user_query,
            primary_profile=primary_profile,
            verbose=verbose,
            context_cfg=context_cfg,
        )
        native_transport = tool_transport_uses_native(
            tool_call_mode=tool_call_mode, primary_profile=primary_profile
        )
        api_messages = messages_for_agent_api_call(messages, agent_system_message or "")
        api_et = frozenset(set(et) | {SECOND_OPINION_TOOL_ID}) if include_second_opinion else et
        response_text = deps.call_ollama_chat(
            api_messages,
            primary_profile,
            api_et,
            verbose=verbose,
            include_second_opinion=include_second_opinion,
        )
        raw_ollama_msg = consume_last_ollama_raw_message()
        response_data = deps.parse_agent_json(response_text)
        note = consume_last_json_parse_note()
        if note:
            sink_emit({"type": "warning", "text": note})
        from agentlib import agent_json as _agent_json_mod
        from agentlib.llm.tool_schemas import tool_transport_label

        tool_transport = _agent_json_mod.consume_last_tool_transport()
        action = response_data.get("action")
        if action == "answer":
            rt = (response_text or "").strip()
            if rt.startswith("{") and ("\"action\"" in rt or "'action'" in rt) and not rt.rstrip().endswith("}"):
                _transcript_append_assistant_user(
                    messages,
                    raw_ollama_msg,
                    response_text,
                    truncated_json_recovery_user_content(
                        tool_call_mode=tool_call_mode, primary_profile=primary_profile
                    ),
                    native_transport=native_transport,
                )
                continue
            if web_required and not saw_strong_web_result:
                _transcript_append_assistant_user(
                    messages,
                    raw_ollama_msg,
                    response_text,
                    (
                        "You must not answer from memory for this request because web verification is required. "
                        "No usable web results have been obtained yet (or they were empty/blocked). "
                        f"Call {mandatory_web_tool} again with a different, more effective query, or fetch_page on a credible source URL "
                        f"from any results you do have. {tool_call_only_nudge(tool_call_mode=tool_call_mode, primary_profile=primary_profile)}"
                    ),
                    native_transport=native_transport,
                )
                continue
            ans_out0 = response_data.get("answer")
            # Refusal-gate: when a model answers with "I can't access..." despite having allowed tools,
            # force one more step that requires a tool call.
            if (
                not web_required
                and not forced_tool_attempt_after_refusal
                and isinstance(ans_out0, str)
                and _answer_refuses_despite_tools(ans_out0)
                and et
            ):
                forced_tool_attempt_after_refusal = True
                _transcript_append_assistant_user(
                    messages,
                    raw_ollama_msg,
                    response_text,
                    (
                        "You have permission to use the allowed tools in this session. "
                        "Do not refuse due to lack of access when a relevant tool exists. "
                        "Pick an appropriate tool and call it now. "
                        f"{tool_call_only_nudge(tool_call_mode=tool_call_mode, primary_profile=primary_profile)}"
                    ),
                    native_transport=native_transport,
                )
                continue
            if web_required and not verified_by_fetch:
                # Web required means we need a real page fetch, not just URL-backed search snippets.
                # However, if the correct response is to ask the user for clarification (ambiguous request),
                # allow that without further tool calls.
                if isinstance(ans_out0, str) and _answer_is_clarifying_question(ans_out0):
                    pass
                else:
                    _transcript_append_assistant_user(
                        messages,
                        raw_ollama_msg,
                        response_text,
                        (
                            "Web verification is required. You must call fetch_page on a credible source URL "
                            "before giving a final answer. If the current results don't include a good URL, "
                            f"call {mandatory_web_tool} again with a better query to find a more direct page, then fetch_page it. "
                            f"Do not answer yet. {tool_call_only_nudge(tool_call_mode=tool_call_mode, primary_profile=primary_profile)}"
                        ),
                        native_transport=native_transport,
                    )
                    continue
            if (
                web_required
                and saw_strong_web_result
                and isinstance(ans_out0, str)
                and _answer_requests_more_web(ans_out0)
            ):
                _transcript_append_assistant_user(
                    messages,
                    raw_ollama_msg,
                    response_text,
                    (
                        "You said you need to search again. Do not give a final answer yet. "
                        f"Call {mandatory_web_tool} again with a more effective query (or fetch_page on a credible URL). "
                        f"{tool_call_only_nudge(tool_call_mode=tool_call_mode, primary_profile=primary_profile)}"
                    ),
                    native_transport=native_transport,
                )
                continue
            if (
                web_required
                and saw_strong_web_result
                and isinstance(ans_out0, str)
                and _answer_deflects_instead_of_verifying(ans_out0)
            ):
                _transcript_append_assistant_user(
                    messages,
                    raw_ollama_msg,
                    response_text,
                    (
                        "Web verification is required and you already have URL-backed search results. "
                        "Do not punt the user to 'visit a website' or say you can't answer from snippets. "
                        "Fetch and verify: call fetch_page on a credible source URL from the results (prefer official sites), "
                        "then answer with the verified name and cite the source. "
                        f"{tool_call_only_nudge(tool_call_mode=tool_call_mode, primary_profile=primary_profile)}"
                    ),
                    native_transport=native_transport,
                )
                continue
            if deliverable_wanted and deliverable_path and not deliverable_read_ok:
                _transcript_append_assistant_user(
                    messages,
                    raw_ollama_msg,
                    response_text,
                    deps.deliverable_followup_block(deliverable_path),
                    native_transport=native_transport,
                )
                continue
            if deliverable_wanted and deliverable_read_ok and deps.answer_missing_written_body(
                response_data.get("answer") or "", deliverable_file_chars
            ):
                _transcript_append_assistant_user(
                    messages,
                    raw_ollama_msg,
                    response_text,
                    deliverable_full_document_user_content(
                        tool_call_mode=tool_call_mode, primary_profile=primary_profile
                    ),
                    native_transport=native_transport,
                )
                continue
            na = (response_data.get("next_action") or "finalize").strip().lower()
            if na == "second_opinion":
                if not include_second_opinion:
                    msg = second_opinion_unavailable_user_content(
                        tool_call_mode=tool_call_mode, primary_profile=primary_profile
                    )
                elif second_opinion_rounds >= 3:
                    msg = second_opinion_limit_user_content(
                        tool_call_mode=tool_call_mode, primary_profile=primary_profile
                    )
                else:
                    msg = (
                        f"Use the {SECOND_OPINION_TOOL_ID} native tool instead of next_action in JSON. "
                        f"Call {SECOND_OPINION_TOOL_ID!r} via tool_calls with draft_answer and rationale."
                    )
                _transcript_append_assistant_user(
                    messages, raw_ollama_msg, response_text, msg, native_transport=native_transport
                )
                continue
            if not reviewed_tool_need and not tool_executed:
                reviewed_tool_need = True
                messages.append(
                    assistant_transcript_message(
                        raw_ollama_msg,
                        fallback_content=response_text,
                        use_provider_shape=native_transport,
                    )
                )
                proposed = response_data.get("answer") or ""
                router_q2 = deps.route_requires_websearch_after_answer(
                    user_query,
                    today_str,
                    proposed,
                    primary_profile,
                    et,
                    transcript_messages=messages,
                )
                if deps.deliverable_skip_mandatory_web(user_query):
                    router_q2 = None
                if router_q2:
                    messages.append(
                        {
                            "role": "user",
                            "content": web_search_required_user_content(
                                mandatory_web_tool,
                                router_q2,
                                tool_call_mode=tool_call_mode,
                                primary_profile=primary_profile,
                                lead_in="Before finalizing",
                            ),
                        }
                    )
                    continue
                if deliverable_wanted:
                    follow = deps.deliverable_first_answer_followup(user_query, proposed)
                elif deps.is_self_capability_question(user_query):
                    follow = deps.self_capability_followup(user_query, proposed)
                else:
                    follow = deps.tool_need_review_followup(user_query, proposed, et)
                messages.append({"role": "user", "content": follow})
                continue
            messages.append(
                assistant_transcript_message(
                    raw_ollama_msg,
                    fallback_content=response_text,
                    use_provider_shape=native_transport,
                )
            )
            ans_out = response_data.get("answer")
            if ans_out is None or (isinstance(ans_out, str) and not ans_out.strip()):
                extracted = deps.extract_json_object_from_text(response_text)
                if extracted:
                    try:
                        recovered = deps.parse_agent_json(extracted)
                        n2 = consume_last_json_parse_note()
                        if n2:
                            sink_emit({"type": "warning", "text": n2})
                    except Exception:
                        recovered = None
                    if isinstance(recovered, dict) and recovered.get("action") == "answer":
                        ra = recovered.get("answer")
                        if isinstance(ra, str) and ra.strip():
                            response_data = recovered
                            ans_out = ra
                if ans_out is None or (isinstance(ans_out, str) and not ans_out.strip()):
                    messages.append(
                        {
                            "role": "user",
                            "content": missing_answer_field_user_content(
                                tool_call_mode=tool_call_mode, primary_profile=primary_profile
                            ),
                        }
                    )
                    continue
            if print_answer and not llm_streaming.assistant_answer_was_streamed():
                sink_emit({"type": "answer", "text": ans_out if ans_out is not None else ""})
            final_answer = ans_out if isinstance(ans_out, str) else str(ans_out)
            answered = True
            break
        elif action == "error":
            messages.append(
                assistant_transcript_message(
                    raw_ollama_msg,
                    fallback_content=response_text,
                    use_provider_shape=native_transport,
                )
            )
            err = response_data.get("error")
            sink_emit({"type": "error", "text": f"Agent error: {err}"})
            final_answer = str(err) if err is not None else None
            answered = True
            break
        elif action == "tool_call" or action in known:
            tool = response_data.get("tool")
            if tool is None:
                tool = action
            params = response_data.get("parameters", {})
            if not isinstance(params, dict):
                params = {}
            params = deps.merge_tool_param_aliases(tool, params)
            params = deps.ensure_tool_defaults(tool, params, user_query)
            params = turn_support.apply_session_cwd_tool_params(tool, params, deps)
            if tool in deps.plugin_tool_handlers:
                params = turn_support.prepare_plugin_tool_browser_params(
                    tool, params, default_engine=deps.default_browser_engine()
                )
            fp = deps.tool_params_fingerprint(tool, params)
            orig_fp = fp
            dedupe_ok = tool not in ("read_file", "tail_file", "grep")
            skipped_duplicate = bool(dedupe_ok and fp in seen_tool_fingerprints)
            policy_blocked = False
            transport_tag = ""
            if tool_transport and tool_transport[0] == tool:
                transport_tag = f" {tool_transport_label(tool, tool_transport[1])}"
            elif tool_transport:
                transport_tag = f" [{tool_transport[1]}]"
            if verbose >= 1:
                if skipped_duplicate:
                    sink_emit({"type": "output", "text": f"[*] Skipping duplicate tool: {tool} (same logical parameters as earlier)"})
                else:
                    if tool == "search_web":
                        sink_emit(
                            {
                                "type": "output",
                                "text": f"[*] Executing tool{transport_tag} ({deps.search_backend_banner_line()}) with {params}",
                            }
                        )
                    else:
                        sink_emit(
                            {
                                "type": "output",
                                "text": f"[*] Executing tool{transport_tag} with {params}",
                            }
                        )
            if skipped_duplicate:
                result = (
                    "[Duplicate call skipped: this tool was already run with the same parameters "
                    "in this session. Use the earlier tool output in this conversation to answer.]"
                )
            else:
                tool_executed = True
                if tool != SECOND_OPINION_TOOL_ID:
                    tool_calls_executed += 1
                if verbose < 1:
                    prog = deps.tool_progress_message(tool, params)
                    if tool_transport and tool_transport[0] == tool:
                        prog = f"{tool_transport_label(tool, tool_transport[1])} {prog}"
                    elif tool_transport:
                        prog = f"[{tool_transport[1]}] {prog}"
                    deps.agent_progress(prog)
                result = ""
                if web_required and tool_calls_executed > mtcw:
                    result = (
                        f"Tool error: web verification budget exceeded ({mtcw} tool calls). "
                        "Stop and explain the verification failure."
                    )
                    policy_blocked = True
                if not policy_blocked and tool == "fetch_page":
                    u_list = turn_support.normalize_fetch_urls(
                        params if isinstance(params, dict) else {},
                        scalar_to_str_fn=deps.scalar_to_str,
                    )
                    n_fetch = len(u_list) if u_list else 1
                    if web_required and fetch_pages_executed + n_fetch > fetch_limit:
                        result = (
                            f"Tool error: web verification budget exceeded ({fetch_limit} fetch_page URL fetches). "
                            "Stop and explain the verification failure."
                        )
                        policy_blocked = True
                    else:
                        fetch_pages_executed += n_fetch
                if policy_blocked:
                    pass
                elif tool in known and tool not in et and tool != SECOND_OPINION_TOOL_ID:
                    policy_blocked = True
                    result = (
                        f"Tool error: {tool} is disabled for this run (tool policy). "
                        "Pick a different allowed tool or respond with action answer."
                    )
                else:
                    try:
                        if tool == SECOND_OPINION_TOOL_ID:
                            if second_opinion_rounds >= 3:
                                result = "second_opinion error: limit reached for this session (max 3)."
                            else:
                                second_opinion_rounds += 1
                                reviewed_tool_need = True
                                result = run_second_opinion_tool(
                                    params,
                                    user_query,
                                    second_opinion_enabled=second_opinion_enabled,
                                    cloud_ai_enabled=cloud_ai_enabled,
                                    reviewer_hosted_profile=reviewer_hosted_profile,
                                    reviewer_ollama_model=reviewer_ollama_model,
                                    hosted_review_ready=deps.hosted_review_ready,
                                    second_opinion_reviewer_messages_fn=deps.second_opinion_reviewer_messages,
                                    call_ollama_plaintext=deps.call_ollama_plaintext,
                                    call_hosted_chat_plain=deps.call_hosted_chat_plain,
                                    call_openai_chat_plain=deps.call_openai_chat_plain,
                                    ollama_second_opinion_model=deps.ollama_second_opinion_model,
                                    scalar_to_str=deps.scalar_to_str,
                                )
                        elif tool == "search_web":
                            result = deps.search_web(params.get("query"), params=params)
                        elif tool == "search_web_fetch_top":
                            result = deps.search_web_fetch_top(params.get("query"), params=params)
                        elif tool == "fetch_page":
                            result = deps.fetch_page(params)
                        elif tool == "run_command":
                            cmd = deps.scalar_to_str(params.get("command"), "")
                            if web_required and re.search(r"\b(curl|wget)\b", cmd):
                                result = (
                                    "Command error: blocked. When web verification is required, do not use run_command "
                                    "with curl/wget to fetch web content. Use fetch_page instead."
                                )
                            else:
                                result = turn_support.run_command_with_session_cwd(deps, cmd)
                        elif tool == "use_git":
                            result = deps.use_git(params)
                        elif tool == "write_file":
                            result = deps.write_file(params.get("path"), params.get("content"))
                        elif tool == "list_directory":
                            result = deps.list_directory(params.get("path"))
                        elif tool == "read_file":
                            result = deps.read_file(params.get("path"))
                        elif tool == "grep":
                            result = deps.grep(
                                params.get("pattern"),
                                params.get("path", "."),
                                params.get("glob_pattern") if params.get("glob_pattern") is not None else params.get("glob"),
                                params.get("max_matches", 200),
                                params.get("max_files", 8000),
                                params.get("ignore_case", False),
                            )
                        elif tool == "download_file":
                            result = deps.download_file(params.get("url"), params.get("path"))
                        elif tool == "tail_file":
                            result = deps.tail_file(params.get("path"), params.get("lines", 20))
                        elif tool == "replace_text":
                            result = deps.replace_text(
                                params.get("path"),
                                params.get("pattern"),
                                params.get("replacement"),
                                params.get("replace_all", True),
                            )
                        elif tool == "call_python":
                            result = deps.call_python(params.get("code"), params.get("globals"))
                        elif tool == "session_command":
                            result = deps.execute_session_command(
                                deps.scalar_to_str(params.get("command"), "")
                            )
                        elif mcp_registry.is_mcp_tool(tool):
                            result = deps.call_mcp_tool(tool, params if isinstance(params, dict) else {})
                        elif tool in deps.plugin_tool_handlers:
                            result = deps.plugin_tool_handlers[tool](params)
                        else:
                            result = f"Unknown tool: {tool}"
                    except KeyboardInterrupt:
                        raise
                    except BaseException as e:
                        result = deps.tool_fault_result(str(tool), e)
            if (
                deps.tool_recovery_may_run(interactive_tool_recovery)
                and not skipped_duplicate
                and not policy_blocked
                and tool in deps.tool_recovery_tools
                and deps.tool_result_indicates_retryable_failure(tool, result)
            ):
                old_params = dict(params)
                sug = deps.suggest_tool_recovery_params(
                    tool,
                    old_params,
                    result,
                    user_query,
                    primary_profile,
                    et,
                    verbose,
                )
                if sug is not None:
                    new_params, rationale = sug
                    new_fp = deps.tool_params_fingerprint(tool, new_params)
                    if new_fp == orig_fp:
                        if verbose >= 1:
                            sink_emit({"type": "output", "text": "[*] Tool recovery: proposed parameters unchanged; skip retry."})
                    elif dedupe_ok and new_fp in seen_tool_fingerprints:
                        if verbose >= 1:
                            sink_emit(
                                {
                                    "type": "output",
                                    "text": "[*] Tool recovery: proposed parameters match an earlier tool call; skip retry.",
                                }
                            )
                    elif deps.confirm_tool_recovery_retry(
                        tool,
                        old_params,
                        new_params,
                        rationale,
                        interactive_tool_recovery=interactive_tool_recovery,
                    ):
                        params = new_params
                        params = turn_support.apply_session_cwd_tool_params(tool, params, deps)
                        fp = new_fp
                        if verbose >= 1:
                            sink_emit({"type": "output", "text": f"[*] Re-running {tool} after confirmed recovery."})
                        else:
                            deps.agent_progress(f"Tool: {tool} (retry)")
                        tool_executed = True
                        if tool == "run_command":
                            cmd = deps.scalar_to_str(params.get("command"), "")
                            if web_required and re.search(r"\b(curl|wget)\b", cmd):
                                result = (
                                    "Command error: blocked. When web verification is required, do not use run_command "
                                    "with curl/wget to fetch web content. Use fetch_page instead."
                                )
                            else:
                                result = turn_support.run_command_with_session_cwd(deps, cmd)
                        elif tool == "call_python":
                            result = deps.call_python(params.get("code"), params.get("globals"))
                        elif tool == "session_command":
                            result = deps.execute_session_command(
                                deps.scalar_to_str(params.get("command"), "")
                            )
                        elif tool == "search_web":
                            result = deps.search_web(params.get("query"), params=params)
                        elif tool == "search_web_fetch_top":
                            result = deps.search_web_fetch_top(params.get("query"), params=params)
                        elif tool == "fetch_page":
                            result = deps.fetch_page(params)
                        note = "[After one user-confirmed corrected retry]\n"
                        if isinstance(result, str) and not result.startswith(
                            "[After one user-confirmed corrected retry]"
                        ):
                            result = note + result
                    elif verbose >= 1:
                        sink_emit({"type": "output", "text": "[*] Tool recovery: retry not confirmed."})
            if tool == "write_file" and deliverable_wanted and not policy_blocked:
                wp = deps.scalar_to_str(params.get("path"), "").strip()
                if wp and (not str(result).startswith("Write error:")):
                    deliverable_path = wp
                    deliverable_read_ok = False
                    deliverable_file_chars = 0
            if tool == "read_file" and deliverable_wanted and deliverable_path and not policy_blocked:
                rp = deps.scalar_to_str(params.get("path"), "").strip()
                if rp == deliverable_path and (not str(result).startswith("Read error:")):
                    deliverable_read_ok = True
                    deliverable_file_chars = len(str(result))
            if not skipped_duplicate and not policy_blocked:
                # "Verified web" can come from either:
                # - search_web returning at least one URL-backed result, OR
                # - fetch_page successfully retrieving a page (the output includes a URL and is not an error).
                #
                # Previously, only search_web could satisfy this latch. In practice, fetch_page is
                # often the actual verification step (search yields thin/blocked results, then fetch
                # provides the authoritative content). Count that as "strong" too.
                if tool == "search_web" and not deps.is_tool_result_weak_for_dedup(result):
                    saw_strong_web_result = True
                elif tool == "search_web_fetch_top" and not deps.is_tool_result_weak_for_dedup(result):
                    saw_strong_web_result = True
                    # This tool includes page fetches + excerpts; count that as verification.
                    rtxt = str(result or "")
                    if rtxt and "[Fetched excerpts]" in rtxt and "Excerpt:" in rtxt and "Fetch error:" not in rtxt[:200]:
                        verified_by_fetch = True
                elif tool == "fetch_page":
                    rtxt = str(result or "")
                    if (
                        rtxt
                        and not rtxt.startswith("Fetch error:")
                        and re.search(r"https?://", rtxt)
                    ):
                        saw_strong_web_result = True
                        verified_by_fetch = True
            if dedupe_ok and not skipped_duplicate and not policy_blocked:
                if orig_fp not in seen_tool_fingerprints:
                    seen_tool_fingerprints.add(orig_fp)
                if fp != orig_fp and fp not in seen_tool_fingerprints:
                    seen_tool_fingerprints.add(fp)
            deliverable_reminder = ""
            if deliverable_wanted and deliverable_path and not deliverable_read_ok:
                deliverable_reminder = (
                    f"Goal reminder (user request): {user_query}\n"
                    + deps.deliverable_followup_block(deliverable_path)
                )
            elif deliverable_wanted and not deliverable_path:
                deliverable_reminder = (
                    f"Goal reminder (user request): {user_query}\n"
                    "If you will satisfy this with write_file, plan to read_file that same path before answering."
                )
            _transcript_append_tool_followup(
                messages,
                raw_ollama_msg,
                response_text,
                tool,
                result,
                deps.tool_result_user_message(
                    tool,
                    params,
                    result,
                    deliverable_reminder=deliverable_reminder,
                    native_transport=native_transport,
                ),
                native_transport=native_transport,
            )
        else:
            _transcript_append_assistant_user(
                messages,
                raw_ollama_msg,
                response_text,
                invalid_agent_response_user_content(
                    tool_call_mode=tool_call_mode,
                    primary_profile=primary_profile,
                ),
                native_transport=native_transport,
            )
            continue
    if not answered:
        if web_required and not saw_strong_web_result:
            sink_emit(
                {
                    "type": "output",
                    "text": (
                        "Unable to verify with web: no strong search result (URL-backed) was obtained in this turn. "
                        "Refusing to answer from memory alone. "
                        "Try again with a more specific query, fetch_page on a URL the user provided, "
                        "or check network / site blocking."
                    ),
                }
            )
        else:
            sink_emit(
                {
                    "type": "output",
                    "text": (
                        "Unable to complete the request within the step limit."
                        if not web_required
                        else (
                            "Unable to complete web verification within the step/tool budget. "
                            "Try again with a more specific query or provide a source URL to fetch."
                        )
                    ),
                }
            )
    return answered, final_answer
