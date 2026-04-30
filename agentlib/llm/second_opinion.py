"""Prompt payloads for independent second-opinion review."""

from __future__ import annotations


def second_opinion_reviewer_messages(user_query: str, primary_answer: str, rationale: str) -> list:
    return [
        {
            "role": "system",
            "content": (
                "You are an independent reviewer. Reply in plain text (not JSON). "
                "Be concise: note agreement, corrections, caveats, or missing checks. "
                "Do not refuse solely because the topic is sensitive—give a substantive review."
            ),
        },
        {
            "role": "user",
            "content": (
                "User request:\n"
                f"{user_query}\n\n"
                "Primary model answer:\n"
                f"{primary_answer}\n\n"
                "The primary model asked for a second opinion for this reason:\n"
                f"{rationale}\n\n"
                "Provide your second opinion."
            ),
        },
    ]


def second_opinion_result_user_message(review_text: str) -> str:
    return (
        "An independent review was obtained. Review text:\n\n"
        f"{review_text}\n\n"
        "Using this review (and earlier context), respond with JSON only. "
        'Typically merge into a single {"action":"answer","answer":"...","next_action":"finalize",'
        '"rationale":"..."} unless you still need tools.'
    )
