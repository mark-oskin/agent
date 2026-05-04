"""Multi-step agent conversation loop (LLM + tools + second opinion)."""

from __future__ import annotations

import re
from typing import AbstractSet, Any, Optional, Tuple

from agentlib.sink import sink_emit
from agentlib.tools.routing import preferred_web_search_tool

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
) -> Tuple[bool, Optional[str]]:
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
    for _ in range(step_limit):
        messages = deps.maybe_compact_context_window(
            messages,
            user_query=user_query,
            primary_profile=primary_profile,
            verbose=verbose,
            context_cfg=context_cfg,
        )
        response_text = deps.call_ollama_chat(
            messages, primary_profile, et, verbose=verbose
        )
        response_data = deps.parse_agent_json(response_text)
        action = response_data.get("action")
        if action == "answer":
            rt = (response_text or "").strip()
            if rt.startswith("{") and ("\"action\"" in rt or "'action'" in rt) and not rt.rstrip().endswith("}"):
                messages.append({"role": "assistant", "content": response_text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your last response looked like a JSON object but it was truncated/malformed "
                            "(missing closing braces/quotes). Respond again with a SINGLE valid JSON object "
                            "and no other text."
                        ),
                    }
                )
                continue
            if web_required and not saw_strong_web_result:
                messages.append({"role": "assistant", "content": response_text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You must not answer from memory for this request because web verification is required. "
                            "No usable web results have been obtained yet (or they were empty/blocked). "
                            f"Call {mandatory_web_tool} again with a different, more effective query, or fetch_page on a credible source URL "
                            "from any results you do have. Respond with JSON tool_call only."
                        ),
                    }
                )
                continue
            ans_out0 = response_data.get("answer")
            if web_required and not verified_by_fetch:
                # Web required means we need a real page fetch, not just URL-backed search snippets.
                # However, if the correct response is to ask the user for clarification (ambiguous request),
                # allow that without further tool calls.
                if isinstance(ans_out0, str) and _answer_is_clarifying_question(ans_out0):
                    pass
                else:
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Web verification is required. You must call fetch_page on a credible source URL "
                                "before giving a final answer. If the current results don't include a good URL, "
                                f"call {mandatory_web_tool} again with a better query to find a more direct page, then fetch_page it. "
                                "Do not answer yet. Respond with JSON tool_call only."
                            ),
                        }
                    )
                    continue
            if (
                web_required
                and saw_strong_web_result
                and isinstance(ans_out0, str)
                and _answer_requests_more_web(ans_out0)
            ):
                messages.append({"role": "assistant", "content": response_text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You said you need to search again. Do not respond with action answer yet. "
                            f"Call {mandatory_web_tool} again with a more effective query (or fetch_page on a credible URL) "
                            "and respond with JSON tool_call only."
                        ),
                    }
                )
                continue
            if (
                web_required
                and saw_strong_web_result
                and isinstance(ans_out0, str)
                and _answer_deflects_instead_of_verifying(ans_out0)
            ):
                messages.append({"role": "assistant", "content": response_text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Web verification is required and you already have URL-backed search results. "
                            "Do not punt the user to 'visit a website' or say you can't answer from snippets. "
                            "Fetch and verify: call fetch_page on a credible source URL from the results (prefer official sites), "
                            "then answer with the verified name and cite the source. Respond with JSON tool_call only."
                        ),
                    }
                )
                continue
            if deliverable_wanted and deliverable_path and not deliverable_read_ok:
                messages.append({"role": "assistant", "content": response_text})
                messages.append(
                    {
                        "role": "user",
                        "content": deps.deliverable_followup_block(deliverable_path),
                    }
                )
                continue
            if deliverable_wanted and deliverable_read_ok and deps.answer_missing_written_body(
                response_data.get("answer") or "", deliverable_file_chars
            ):
                messages.append({"role": "assistant", "content": response_text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your answer is too short to be the requested multi-page document. "
                            "Use read_file to load the written file, then respond with action answer whose "
                            "answer field contains the FULL document text (the user asked for the document itself). "
                            "If the file is still too short, expand it with write_file and read_file again."
                        ),
                    }
                )
                continue
            na = (response_data.get("next_action") or "finalize").strip().lower()
            if na == "second_opinion":
                rationale = deps.scalar_to_str(response_data.get("rationale"), "").strip()
                primary = response_data.get("answer") or ""
                if not rationale:
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "second_opinion requires a non-empty string field \"rationale\" explaining why "
                                "you want a review. Respond with JSON only."
                            ),
                        }
                    )
                    continue
                hosted_ready = deps.hosted_review_ready(
                    cloud_ai_enabled, reviewer_hosted_profile
                )
                if not second_opinion_enabled and not hosted_ready:
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Second opinion is not available in this session. Respond with JSON only using "
                                '{"action":"answer","answer":"...","next_action":"finalize","rationale":"..."}.'
                            ),
                        }
                    )
                    continue
                backend = (
                    "ollama"
                    if second_opinion_enabled
                    else ("openai" if hosted_ready else "")
                )
                if second_opinion_rounds >= 3:
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Second opinion limit reached for this session. Respond with JSON only using "
                                '{"action":"answer","answer":"...","next_action":"finalize","rationale":"..."}.'
                            ),
                        }
                    )
                    continue
                reviewer_msgs = deps.second_opinion_reviewer_messages(
                    user_query, primary, rationale
                )
                if backend == "ollama":
                    rm = (reviewer_ollama_model or "").strip() or deps.ollama_second_opinion_model()
                    review = deps.call_ollama_plaintext(reviewer_msgs, rm)
                else:
                    if (
                        reviewer_hosted_profile is not None
                        and reviewer_hosted_profile.backend == "hosted"
                        and (reviewer_hosted_profile.api_key or "").strip()
                    ):
                        review = deps.call_hosted_chat_plain(
                            reviewer_msgs, reviewer_hosted_profile
                        )
                    else:
                        review = deps.call_openai_chat_plain(reviewer_msgs)
                second_opinion_rounds += 1
                tool_executed = True
                reviewed_tool_need = True
                messages.append({"role": "assistant", "content": response_text})
                messages.append(
                    {
                        "role": "user",
                        "content": deps.second_opinion_result_user_message(review),
                    }
                )
                continue
            if not reviewed_tool_need and not tool_executed:
                reviewed_tool_need = True
                messages.append({"role": "assistant", "content": response_text})
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
                            "content": (
                                f"Before finalizing, you MUST call the tool {mandatory_web_tool} to verify.\n"
                                "Respond with JSON only in tool_call form.\n"
                                f'Suggested query: "{router_q2}"'
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
            messages.append({"role": "assistant", "content": response_text})
            ans_out = response_data.get("answer")
            if ans_out is None or (isinstance(ans_out, str) and not ans_out.strip()):
                extracted = deps.extract_json_object_from_text(response_text)
                if extracted:
                    try:
                        recovered = deps.parse_agent_json(extracted)
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
                            "content": (
                                'Your JSON had action "answer" but was missing a non-empty string field "answer". '
                                "Respond again with a SINGLE valid JSON object in this exact shape:\n"
                                '{"action":"answer","answer":"..."}\n'
                                "No other keys, and no other text."
                            ),
                        }
                    )
                    continue
            if print_answer:
                sink_emit({"type": "answer", "text": ans_out if ans_out is not None else ""})
            final_answer = ans_out if isinstance(ans_out, str) else str(ans_out)
            answered = True
            break
        elif action == "error":
            messages.append({"role": "assistant", "content": response_text})
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
            fp = deps.tool_params_fingerprint(tool, params)
            orig_fp = fp
            dedupe_ok = tool not in ("read_file", "tail_file")
            skipped_duplicate = bool(dedupe_ok and fp in seen_tool_fingerprints)
            policy_blocked = False
            if verbose >= 1:
                if skipped_duplicate:
                    sink_emit({"type": "output", "text": f"[*] Skipping duplicate tool: {tool} (same logical parameters as earlier)"})
                else:
                    if tool == "search_web":
                        sink_emit(
                            {
                                "type": "output",
                                "text": f"[*] Executing tool: {tool} ({deps.search_backend_banner_line()}) with {params}",
                            }
                        )
                    else:
                        sink_emit({"type": "output", "text": f"[*] Executing tool: {tool} with {params}"})
            if skipped_duplicate:
                result = (
                    "[Duplicate call skipped: this tool was already run with the same parameters "
                    "in this session. Use the earlier tool output in this conversation to answer.]"
                )
            else:
                tool_executed = True
                tool_calls_executed += 1
                if verbose < 1:
                    deps.agent_progress(deps.tool_progress_message(tool, params))
                result = ""
                if web_required and tool_calls_executed > mtcw:
                    result = (
                        f"Tool error: web verification budget exceeded ({mtcw} tool calls). "
                        "Stop and explain the verification failure."
                    )
                    policy_blocked = True
                if not policy_blocked and tool == "fetch_page":
                    fetch_pages_executed += 1
                    if web_required and fetch_pages_executed > fetch_limit:
                        result = (
                            f"Tool error: web verification budget exceeded ({fetch_limit} fetch_page calls). "
                            "Stop and explain the verification failure."
                        )
                        policy_blocked = True
                if policy_blocked:
                    pass
                elif tool in known and tool not in et:
                    policy_blocked = True
                    result = (
                        f"Tool error: {tool} is disabled for this run (tool policy). "
                        "Pick a different allowed tool or respond with action answer."
                    )
                else:
                    try:
                        if tool == "search_web":
                            result = deps.search_web(params.get("query"), params=params)
                        elif tool == "search_web_fetch_top":
                            result = deps.search_web_fetch_top(params.get("query"), params=params)
                        elif tool == "fetch_page":
                            result = deps.fetch_page(params.get("url"))
                        elif tool == "run_command":
                            cmd = deps.scalar_to_str(params.get("command"), "")
                            if web_required and re.search(r"\b(curl|wget)\b", cmd):
                                result = (
                                    "Command error: blocked. When web verification is required, do not use run_command "
                                    "with curl/wget to fetch web content. Use fetch_page instead."
                                )
                            else:
                                result = deps.run_command(cmd)
                        elif tool == "use_git":
                            result = deps.use_git(params)
                        elif tool == "write_file":
                            result = deps.write_file(params.get("path"), params.get("content"))
                        elif tool == "list_directory":
                            result = deps.list_directory(params.get("path"))
                        elif tool == "read_file":
                            result = deps.read_file(params.get("path"))
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
                                result = deps.run_command(cmd)
                        elif tool == "call_python":
                            result = deps.call_python(params.get("code"), params.get("globals"))
                        elif tool == "search_web":
                            result = deps.search_web(params.get("query"), params=params)
                        elif tool == "search_web_fetch_top":
                            result = deps.search_web_fetch_top(params.get("query"), params=params)
                        elif tool == "fetch_page":
                            result = deps.fetch_page(params.get("url"))
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
            messages.append({"role": "assistant", "content": response_text})
            messages.append(
                {
                    "role": "user",
                    "content": deps.tool_result_user_message(
                        tool, params, result, deliverable_reminder=deliverable_reminder
                    ),
                }
            )
        else:
            messages.append({"role": "assistant", "content": response_text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your last message was not valid agent JSON. "
                        "Respond with JSON only and include a non-null string action. "
                        'Use {"action":"tool_call","tool":<one of the allowed tools>,'
                        '"parameters":{...}} or {"action":"answer","answer":"..."}.'
                    ),
                }
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
