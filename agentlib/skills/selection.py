from __future__ import annotations

from typing import AbstractSet, Callable, Optional, Tuple


def match_skill_detail(
    user_text: str, skills: Optional[dict]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Trigger-based skill match: pick the skill whose trigger substring is longest match in user_text.
    Returns (skill_id, winning_trigger) or (None, None).
    """
    if not user_text or not skills:
        return None, None
    low = (user_text or "").lower()
    best_sid: Optional[str] = None
    best_tr: Optional[str] = None
    best_len = 0
    for sid, data in skills.items():
        d = data if isinstance(data, dict) else {}
        for tr in d.get("triggers") or [sid]:
            t = (str(tr) or "").lower().strip()
            if not t:
                continue
            if t in low and len(t) > best_len:
                best_len = len(t)
                best_sid = str(sid)
                best_tr = t
    return best_sid, best_tr


def match_skill_id(user_text: str, skills: Optional[dict]) -> Optional[str]:
    sid, _ = match_skill_detail(user_text, skills)
    return sid


def format_skills_for_selector(skills_map: dict) -> str:
    lines = []
    for sid in sorted(skills_map.keys()):
        rec = skills_map.get(sid) or {}
        desc = (rec.get("description") or "").strip() if isinstance(rec, dict) else ""
        w = (rec.get("workflow") or {}) if isinstance(rec, dict) else {}
        multi = bool(isinstance(w, dict) and w)
        has_art = False
        if isinstance(rec, dict):
            has_art = bool(
                (isinstance(rec.get("reference_files"), list) and rec.get("reference_files"))
                or (isinstance(rec.get("doc_urls"), list) and rec.get("doc_urls"))
                or (
                    isinstance(rec.get("grounding_commands"), list)
                    and rec.get("grounding_commands")
                )
            )
        lines.append(
            f"- id: {sid}\n  description: {desc}\n"
            f"  supports_multi_step: {str(multi).lower()}\n"
            f"  has_bundled_grounding: {str(has_art).lower()}"
        )
    return "\n".join(lines)


def ml_select_skill_id(
    user_request: str,
    skills_map: dict,
    *,
    primary_profile,
    verbose: int,
    call_llm_json_content: Callable[..., str],
    agent_progress: Callable[[str], None],
    try_json_loads_object: Callable[[str], object],
    parse_json_with_skill_id: Callable[[str], Optional[dict]],
) -> Tuple[Optional[str], str]:
    """
    Ask the current primary model to choose the best skill id for the request.
    Returns (skill_id or None, rationale).
    """
    if not isinstance(skills_map, dict) or not skills_map:
        return None, "No skills loaded."
    req = (user_request or "").strip()
    if not req:
        return None, "Request is empty."
    skill_listing = format_skills_for_selector(skills_map)
    selector_sys = (
        "You are a skill selector for a coding assistant.\n"
        "Given the user request and the available skills, pick the single best skill id.\n"
        "Respond with JSON only. No Markdown, no code fences.\n"
        'Output schema: {"skill_id":"<id or empty>","rationale":"short"}\n'
        "- Use exactly those two keys. Do not use action, tool, or answer.\n"
        "- Choose exactly one skill_id from the list.\n"
        "- If none are suitable, set skill_id to empty string.\n"
    )
    msgs = [
        {"role": "system", "content": selector_sys},
        {
            "role": "user",
            "content": (
                f"User request:\n{req}\n\n"
                f"Available skills:\n{skill_listing}\n\n"
                "Pick the best skill_id."
            ),
        },
    ]
    agent_progress("Selecting skill (model)…")
    raw = call_llm_json_content(msgs, primary_profile, verbose=verbose)
    first = try_json_loads_object(raw)
    if isinstance(first, dict) and first.get("_call_error"):
        return None, str(first.get("_call_error") or "LLM call failed.")
    obj = parse_json_with_skill_id(raw)
    if not isinstance(obj, dict) or "skill_id" not in obj:
        return None, "Model did not return valid JSON with skill_id."
    sid = (obj.get("skill_id") or "").strip() if isinstance(obj, dict) else ""
    rat = (obj.get("rationale") or "").strip() if isinstance(obj, dict) else ""
    if not sid:
        return None, rat or "Model did not select a skill."
    if sid not in skills_map:
        return None, (rat + " " if rat else "") + f"Model selected unknown skill {sid!r}."
    return sid, rat or f"Selected {sid!r}."

