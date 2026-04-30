"""Multi-step agent conversation loop (LLM + tools + second opinion)."""

from __future__ import annotations

import re
from typing import AbstractSet, Any, Optional, Tuple

from .deps import ConversationTurnDeps


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
) -> Tuple[bool, Optional[str]]:
    et = deps.coerce_enabled_tools(enabled_tools)
    if web_required and "search_web" not in et:
        web_required = False
    seen_tool_fingerprints: set = set()
    reviewed_tool_need = False
    saw_strong_web_result = False
    answered = False
    tool_executed = False
    second_opinion_rounds = 0
    final_answer: Optional[str] = None
    deliverable_path: Optional[str] = None
    deliverable_read_ok = False
    deliverable_file_chars = 0
    known = deps.all_known_tools()
    for _ in range(30):
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
                            "Call search_web again with a different, more effective query, or fetch_page on a credible source URL "
                            "from any results you do have. Respond with JSON tool_call only."
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
                                "Before finalizing, you MUST call the tool search_web to verify.\n"
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
                    follow = deps.tool_need_review_followup(user_query, proposed)
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
                print(ans_out if ans_out is not None else "")
            final_answer = ans_out if isinstance(ans_out, str) else str(ans_out)
            answered = True
            break
        elif action == "error":
            messages.append({"role": "assistant", "content": response_text})
            err = response_data.get("error")
            print(f"Agent error: {err}")
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
                    print(f"[*] Skipping duplicate tool: {tool} (same logical parameters as earlier)")
                else:
                    if tool == "search_web":
                        print(
                            f"[*] Executing tool: {tool} ({deps.search_backend_banner_line()}) with {params}"
                        )
                    else:
                        print(f"[*] Executing tool: {tool} with {params}")
            if skipped_duplicate:
                result = (
                    "[Duplicate call skipped: this tool was already run with the same parameters "
                    "in this session. Use the earlier tool output in this conversation to answer.]"
                )
            else:
                tool_executed = True
                if verbose < 1:
                    deps.agent_progress(deps.tool_progress_message(tool, params))
                result = ""
                if tool in known and tool not in et:
                    policy_blocked = True
                    result = (
                        f"Tool error: {tool} is disabled for this run (tool policy). "
                        "Pick a different allowed tool or respond with action answer."
                    )
                else:
                    try:
                        if tool == "search_web":
                            result = deps.search_web(params.get("query"), params=params)
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
                            print("[*] Tool recovery: proposed parameters unchanged; skip retry.")
                    elif dedupe_ok and new_fp in seen_tool_fingerprints:
                        if verbose >= 1:
                            print(
                                "[*] Tool recovery: proposed parameters match an earlier "
                                "tool call; skip retry."
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
                            print(f"[*] Re-running {tool} after confirmed recovery.")
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
                        elif tool == "fetch_page":
                            result = deps.fetch_page(params.get("url"))
                        note = "[After one user-confirmed corrected retry]\n"
                        if isinstance(result, str) and not result.startswith(
                            "[After one user-confirmed corrected retry]"
                        ):
                            result = note + result
                    elif verbose >= 1:
                        print("[*] Tool recovery: retry not confirmed.")
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
            if (
                tool == "search_web"
                and not skipped_duplicate
                and not policy_blocked
                and not deps.is_tool_result_weak_for_dedup(result)
            ):
                saw_strong_web_result = True
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
            print(
                "Unable to verify with web: no strong search result (URL-backed) was obtained in this turn. "
                "Refusing to answer from memory alone. "
                "Try again with a more specific query, fetch_page on a URL the user provided, "
                "or check network / site blocking."
            )
        else:
            print("Unable to complete the request within the step limit.")
    return answered, final_answer
