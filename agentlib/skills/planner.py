from __future__ import annotations

from typing import AbstractSet, Callable, Optional, Tuple


def skill_plan_steps(
    *,
    user_request: str,
    today_str: str,
    skill_id: str,
    skills_map: dict,
    primary_profile,
    enabled_tools: AbstractSet[str],
    verbose: int,
    system_prompt_override: Optional[str],
    agent_progress: Callable[[str], None],
    call_llm_json_content: Callable[..., str],
    try_json_loads_object: Callable[[str], object],
    parse_workflow_plan_dict: Callable[[str], Optional[dict]],
    scalar_to_int: Callable[[object, int], int],
    scalar_to_str: Callable[[object, str], str],
) -> Tuple[Optional[list], str]:
    """
    If the selected skill declares a workflow, ask the model for a step plan.
    Returns (steps or None, planner_raw_text_or_error).
    """
    # enabled_tools / system_prompt_override are intentionally unused here: the planner
    # is isolated from the main agent and must not see the long tool/JSON contract.
    _ = enabled_tools
    _ = system_prompt_override

    rec = skills_map.get(skill_id) or {}
    wf = (rec.get("workflow") or {}) if isinstance(rec, dict) else {}
    if not isinstance(wf, dict) or not wf:
        return None, "Skill has no workflow."
    planner = (wf.get("planner_prompt") or "").strip()
    if not planner:
        return None, "Skill workflow missing planner_prompt."
    max_steps = scalar_to_int(wf.get("max_steps"), 8)
    if max_steps < 1:
        max_steps = 8
    max_steps = min(max_steps, 20)
    skill_prompt = (rec.get("prompt") or "").strip() if isinstance(rec, dict) else ""
    plan_sys = (
        "You are a workflow planner, not the main coding agent.\n"
        "Reply with a single JSON object. No Markdown, no code fences, no action/answer format.\n"
        "Forbidden top-level keys: action, tool, tool_call, parameters, answer, next_action.\n"
        "Required top-level shape:\n"
        '{"questions":[],"steps":[{"title":"string","details":"string","success":"string"}]}\n'
        f"- questions: optional; may be [].\n"
        f"- steps: at least 1, at most {max_steps}. Each title must be non-empty.\n"
    )
    if skill_prompt:
        plan_sys += "\n--- Skill context ---\n" + skill_prompt + "\n"
    plan_sys += (
        "\n--- Planner instructions ---\n"
        + planner
        + f"\n\nContext: today's date (system clock) is {today_str}.\n"
        "If the user is vague, list questions and still provide a best-guess step plan (do not block on clarification)."
    )
    user_body = f"User request:\n{user_request}"
    msgs: list = [
        {"role": "system", "content": plan_sys},
        {"role": "user", "content": user_body},
    ]
    agent_progress("Planning workflow (model)…")
    raw = call_llm_json_content(msgs, primary_profile, verbose=verbose)
    err0 = try_json_loads_object(raw)
    if isinstance(err0, dict) and err0.get("_call_error"):
        return None, str(err0.get("_call_error") or "Planner call failed.")
    plan_obj = parse_workflow_plan_dict(raw)
    if plan_obj is None and (raw or "").strip():
        repair = (
            "Your last reply was not a valid plan JSON (must include a non-empty \"steps\" array "
            "with {title, details, success} objects). Do not use action, answer, or tool. "
            "Output ONE json object only. Previous output:\n"
        )
        cap = 3200
        repair += (raw[:cap] + ("…" if len(raw) > cap else ""))
        msgs2 = list(msgs) + [{"role": "user", "content": repair}]
        agent_progress("Re-asking model for valid step plan…")
        raw2 = call_llm_json_content(msgs2, primary_profile, verbose=verbose)
        err1 = try_json_loads_object(raw2)
        if isinstance(err1, dict) and err1.get("_call_error"):
            return None, str(err1.get("_call_error") or "Planner retry failed.")
        plan_obj = parse_workflow_plan_dict(raw2)
        if plan_obj is not None:
            raw = raw2
    if plan_obj is None:
        return None, raw
    steps = plan_obj.get("steps")
    if not isinstance(steps, list) or not steps:
        return None, raw
    out_steps = []
    for st in steps:
        if not isinstance(st, dict):
            continue
        title = scalar_to_str(st.get("title"), "").strip()
        details = scalar_to_str(st.get("details"), "").strip()
        success = scalar_to_str(st.get("success"), "").strip()
        if not title:
            continue
        out_steps.append({"title": title, "details": details, "success": success})
        if len(out_steps) >= max_steps:
            break
    return out_steps or None, raw

