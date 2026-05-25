"""Second opinion tool and reviewer prompts."""

from __future__ import annotations

from typing import Any, Callable, Optional


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


def second_opinion_result_user_message(review_text: str, *, native_transport: bool = False) -> str:
    if native_transport:
        return (
            "An independent review was obtained. Review text:\n\n"
            f"{review_text}\n\n"
            "Using this review (and earlier context), reply with plain text as your final answer "
            "unless you still need another tool."
        )
    return (
        "An independent review was obtained. Review text:\n\n"
        f"{review_text}\n\n"
        "Using this review (and earlier context), respond with JSON only. "
        'Typically merge into a single {"action":"answer","answer":"...","next_action":"finalize",'
        '"rationale":"..."} unless you still need tools.'
    )


def run_second_opinion_tool(
    params: dict,
    user_query: str,
    *,
    second_opinion_enabled: bool,
    cloud_ai_enabled: bool,
    reviewer_hosted_profile: Any,
    reviewer_ollama_model: Optional[str],
    hosted_review_ready: Callable[[bool, Any], bool],
    second_opinion_reviewer_messages_fn: Callable[[str, str, str], list],
    call_ollama_plaintext: Callable[[list, str], str],
    call_hosted_chat_plain: Callable[[list, Any], str],
    call_openai_chat_plain: Callable[[list], str],
    ollama_second_opinion_model: Callable[[], str],
    scalar_to_str: Callable[..., str],
) -> str:
    """Execute the ``second_opinion`` native/JSON tool."""
    p = params if isinstance(params, dict) else {}
    draft = scalar_to_str(p.get("draft_answer") or p.get("answer"), "").strip()
    rationale = scalar_to_str(p.get("rationale"), "").strip()
    if not draft:
        return "second_opinion error: missing required parameter draft_answer"
    if not rationale:
        return "second_opinion error: missing required parameter rationale"

    hosted_ready = hosted_review_ready(cloud_ai_enabled, reviewer_hosted_profile)
    if not second_opinion_enabled and not hosted_ready:
        return "second_opinion error: not available in this session (enable second_opinion or configure a hosted reviewer)"

    backend = "ollama" if second_opinion_enabled else ("openai" if hosted_ready else "")
    reviewer_msgs = second_opinion_reviewer_messages_fn(user_query, draft, rationale)
    if backend == "ollama":
        rm = (reviewer_ollama_model or "").strip() or ollama_second_opinion_model()
        review = call_ollama_plaintext(reviewer_msgs, rm)
    elif (
        reviewer_hosted_profile is not None
        and getattr(reviewer_hosted_profile, "backend", "") == "hosted"
        and (getattr(reviewer_hosted_profile, "api_key", "") or "").strip()
    ):
        review = call_hosted_chat_plain(reviewer_msgs, reviewer_hosted_profile)
    else:
        review = call_openai_chat_plain(reviewer_msgs)

    return (
        "Independent review:\n\n"
        f"{review}\n\n"
        "Incorporate this review into your next reply to the user."
    )
