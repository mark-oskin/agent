"""LLM-assisted routing: decide whether a web search should run before or after an answer."""

from __future__ import annotations

from typing import AbstractSet, Callable, Optional


ROUTER_INSTRUCTIONS = (
    "You are a routing assistant for a tool-using agent.\n"
    "Decide whether the user's request requires a web search BEFORE answering.\n"
    "Respond ONLY with JSON. No prose, no Markdown.\n"
    'Output exactly one of:\n'
    '1) {"action":"web_search","query":"..."}  (query must be a non-empty string)\n'
    '2) {"action":"no_web"}\n'
    "\n"
    "Bias rule (IMPORTANT): WHEN IN DOUBT, CHOOSE web_search.\n"
    "\n"
    "You MUST choose web_search for requests that involve:\n"
    "- Anything that changes over time: current events, news, prices, rankings, outages, elections, sports.\n"
    "- Real-world entities/roles: who holds an office/title (CEO, president, etc.), leadership, staffing, ownership.\n"
    "- Software/library/tooling facts that drift: latest versions, APIs, documentation, best practices, defaults, "
    "security guidance, deprecations, release dates.\n"
    "- Comparisons/recommendations likely to be time-dependent: \"best\", \"top\", \"recommended\", \"popular\", "
    "\"vs\" for products/tools.\n"
    "\n"
    "Choose no_web ONLY for clearly timeless content:\n"
    "- Definitions and explanations of stable concepts (math, basic CS concepts).\n"
    "- Pure transformations on user-provided text/data.\n"
    "- Established historical facts explicitly anchored to the past by the user.\n"
)


def router_transcript_slice(transcript_messages: Optional[list], *, router_transcript_max_messages: int) -> list:
    """Last N user/assistant/system messages for routing (bounded for prompt size)."""
    if not transcript_messages:
        return []
    lim = max(1, int(router_transcript_max_messages))
    slice_ = (
        transcript_messages[-lim:]
        if len(transcript_messages) > lim
        else transcript_messages
    )
    out: list = []
    for m in slice_:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip().lower()
        if role in ("user", "assistant", "system"):
            out.append(dict(m))
    return out


def router_llm_messages(transcript_slice: list, tail_user_content: str) -> list:
    if not transcript_slice:
        return [{"role": "user", "content": tail_user_content}]
    return transcript_slice + [{"role": "user", "content": tail_user_content}]


def router_prompt(user_query: str, today_str: str, *, has_prior_transcript: bool = False) -> str:
    uq = (user_query or "").strip()
    hint = ""
    if has_prior_transcript:
        hint = (
            "\nEarlier messages in this chat are relevant. If the latest user message is a short "
            "follow-up (pronouns like they / the game / yesterday), resolve it using that transcript "
            "unless that is impossible.\n"
        )
    return (
        f"{ROUTER_INSTRUCTIONS}\n\n"
        f"Today's date (system clock): {today_str}\n\n"
        f"User request: {uq}\n"
        f"{hint}"
    )


def route_requires_websearch(
    user_query: str,
    today_str: str,
    primary_profile,
    enabled_tools: Optional[AbstractSet[str]],
    transcript_messages: Optional[list],
    *,
    coerce_enabled_tools: Callable[[Optional[AbstractSet[str]]], AbstractSet[str]],
    call_ollama_chat: Callable[..., str],
    parse_agent_json: Callable[[str], dict],
    scalar_to_str: Callable[..., str],
    router_transcript_max_messages: int,
) -> Optional[str]:
    """
    Ask the model whether to do web search first.
    Returns a query string if web search is needed, else None.
    """
    if "search_web" not in coerce_enabled_tools(enabled_tools):
        return None
    slice_ = router_transcript_slice(
        transcript_messages, router_transcript_max_messages=router_transcript_max_messages
    )
    tail = router_prompt(user_query, today_str, has_prior_transcript=bool(slice_))
    msgs = router_llm_messages(slice_, tail)
    try:
        raw = call_ollama_chat(msgs, primary_profile, enabled_tools)
        d = parse_agent_json(raw)
        a = (d.get("action") or "").strip()
        if a == "web_search":
            q = scalar_to_str(d.get("query"), "").strip()
            return q if q else (user_query or "").strip()
        return None
    except Exception:
        return None


def route_requires_websearch_after_answer(
    user_query: str,
    today_str: str,
    proposed_answer: str,
    primary_profile,
    enabled_tools: Optional[AbstractSet[str]],
    transcript_messages: Optional[list],
    *,
    coerce_enabled_tools: Callable[[Optional[AbstractSet[str]]], AbstractSet[str]],
    call_ollama_chat: Callable[..., str],
    parse_agent_json: Callable[[str], dict],
    scalar_to_str: Callable[..., str],
    router_transcript_max_messages: int,
) -> Optional[str]:
    """
    Backup router pass when the model answered tool-free.
    This prompt is intentionally conservative: if verifying would be helpful, search.
    """
    if "search_web" not in coerce_enabled_tools(enabled_tools):
        return None
    uq = (user_query or "").strip()
    ans = (proposed_answer or "").strip()
    slice_ = router_transcript_slice(
        transcript_messages, router_transcript_max_messages=router_transcript_max_messages
    )
    prior = ""
    if slice_:
        prior = (
            "\nSession context: earlier messages may define what the user meant; align verification "
            "searches with that topic when the latest request is a follow-up.\n"
        )
    prompt = (
        f"{ROUTER_INSTRUCTIONS}\n\n"
        "Extra guidance: You are reviewing an already-drafted answer. "
        "If the answer includes any real-world factual claim that could be outdated or wrong, choose web_search.\n\n"
        f"Today's date (system clock): {today_str}\n\n"
        f"User request: {uq}\n\n"
        f"Proposed answer: {ans}\n"
        f"{prior}"
    )
    msgs = router_llm_messages(slice_, prompt)
    try:
        raw = call_ollama_chat(msgs, primary_profile, enabled_tools)
        d = parse_agent_json(raw)
        a = (d.get("action") or "").strip()
        if a == "web_search":
            q = scalar_to_str(d.get("query"), "").strip()
            return q if q else uq
        return None
    except Exception:
        return None
