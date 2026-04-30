"""Follow-up prompts for router edge cases (tool-free first answers, capability asks)."""

from __future__ import annotations

import re


def tool_need_review_followup(user_query: str, proposed_answer: str) -> str:
    """
    Model-driven check when the assistant answered tool-free on the first turn.

    The old wording invited models to discuss "timeless vs current" in the answer field instead
    of answering the user; the user only ever sees the `answer` string.
    """
    uq = (user_query or "").strip()
    ans = (proposed_answer or "").strip()
    return (
        "You just responded with action=answer without using any tools.\n\n"
        "User request:\n"
        f"{uq}\n\n"
        "Your proposed answer (the user will NOT see this self-review—only your NEXT JSON matters):\n"
        f"{ans}\n\n"
        "Internally decide: does answering the user's question correctly require fresh, verifiable "
        "facts from the web (news, prices, who holds an office today, versions, outages, etc.)?\n\n"
        "Respond with JSON only:\n"
        '- If YES, use {"action":"tool_call","tool":"search_web","parameters":{"query":"..."}} with a focused query.\n'
        '- If NO, use {"action":"answer","answer":"..."} where the `answer` value is your **complete** reply to the '
        "user's question—normal helpful content only. Do **not** fill `answer` with meta about whether web search "
        'is needed, "timeless" facts, or timeliness—those belong in your internal decision, not in `answer`.\n'
        "If unsure whether facts may be stale, prefer search_web.\n"
        "Do not include any other keys."
    )


def is_self_capability_question(user_query: str) -> bool:
    """Questions about the assistant itself (not third-party facts) — generic web-review misfires."""
    q = (user_query or "").strip().lower()
    if not q:
        return False
    return bool(
        re.search(
            r"\b(what\s+(kind\s+of\s+)?model|which\s+model|what\s+llm|"
            r"who\s+are\s+you|what\s+are\s+you|your\s+capabilities|"
            r"what\s+can\s+you\s+do|what\s+do\s+you\s+support|what\s+tools\s+do\s+you|"
            r"describe\s+yourself|your\s+limitations|"
            r"what\s+kinds?\s+of\s+(outputs?|inputs?)|\binputs?\s+and\s+outputs?)\b",
            q,
        )
    )


def self_capability_followup(user_query: str, proposed_answer: str) -> str:
    """Replace generic web-vs-memory wording for identity/capability asks."""
    uq = (user_query or "").strip()
    ans = (proposed_answer or "").strip()
    tools = (
        "search_web, fetch_page, run_command, use_git, write_file, read_file, list_directory, "
        "download_file, tail_file, replace_text, call_python"
    )
    return (
        "The user is asking about **this assistant**: what model/setup it is, what it can accept or "
        "produce, and/or what tools exist in this agent—not for a lecture on web search, timeliness, "
        'or "timeless" vs current facts.\n\n'
        f"User request:\n{uq}\n\n"
        f"Your last `answer` was not what they asked for (do not repeat this pattern):\n{ans}\n\n"
        "Respond with JSON only:\n"
        '{"action":"answer","answer":"..."}\n'
        "The `answer` string must **directly** address their question: plain-language description of "
        "what you are (as far as this session's context allows), that interaction here is JSON "
        f"tool/answer messages, and the concrete tools available in this script ({tools}). "
        "No preamble about whether web search is required."
    )
