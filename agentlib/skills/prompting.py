from __future__ import annotations

from typing import Optional


def apply_skill_prompt_overlay(
    system_instruction_text: str, *, skill_id: Optional[str], skills_map: Optional[dict]
) -> str:
    """
    Apply the selected skill's prompt as an overlay to the system instruction text.
    Keeps the exact banner formatting stable for tests and user-facing output.
    """
    si = system_instruction_text or ""
    if not skill_id or not skills_map or not isinstance(skills_map, dict):
        return si
    rec0 = skills_map.get(skill_id) or {}
    suff = str(rec0.get("prompt") or "").strip() if isinstance(rec0, dict) else ""
    if not suff:
        return si
    return si + "\n\n--- Active skill ---\n" + suff

