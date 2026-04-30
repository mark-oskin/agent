"""Parse and judge `/while` REPL commands."""

from __future__ import annotations

from typing import Callable, Tuple

WHILE_JUDGE_SYSTEM = (
    "You are a strict boolean judge for a /while command in a coding assistant.\n"
    "The human gave a natural-language /while CONDITION (like C: while (CONDITION) { body }).\n"
    "Decide whether that CONDITION is TRUE or FALSE right now, using ONLY the conversation excerpt below.\n"
    "Reply with EXACTLY one character and nothing else: 0 or 1.\n"
    "Meaning (must follow this mapping):\n"
    "- 1 = the condition is TRUE — the /while should KEEP GOING (run the `do` body, then re-check).\n"
    "- 0 = the condition is FALSE — the /while should EXIT (do not run the body; stop the loop).\n"
    "Do not output markdown, words, explanations, or whitespace around the digit."
)


def while_conversation_excerpt_for_judge(
    messages: list,
    max_chars: int = 12000,
    *,
    scalar_to_str_fn: Callable[..., str],
) -> str:
    """Compact transcript text for condition judging (truncate per message)."""
    if not messages:
        return "(empty conversation)"
    chunks: list[str] = []
    remaining = max_chars
    per_cap = max(400, max_chars // max(6, len(messages)))
    for m in messages[-40:]:
        role = str((m or {}).get("role") or "?")
        content = scalar_to_str_fn((m or {}).get("content"), "")[:per_cap]
        piece = f"[{role}]:\n{content}"
        if len(piece) > remaining:
            piece = piece[: max(0, remaining - 1)] + "…"
        chunks.append(piece)
        remaining -= len(piece) + 2
        if remaining <= 0:
            break
    body = "\n\n".join(chunks)
    if len(body) > max_chars:
        body = body[: max_chars - 1] + "…"
    return body


def parse_while_judge_bit(text: str) -> int:
    """Extract first 0 or 1 from model output; default 0 (false / exit) if ambiguous."""
    t = (text or "").strip()
    if not t:
        return 0
    for chunk in (t.splitlines()[0], t):
        for c in chunk.strip():
            if c == "1":
                return 1
            if c == "0":
                return 0
    return 0


def post_do_tokens_to_body_prompts(post_do_tokens: list[str]) -> list[str]:
    """
    Build prompt strings after `do`. Commas separate prompts.

    shlex often attaches a comma to the previous token (`"p1", "p2"` -> `p1,`, `p2`).
    Split those so comma becomes its own delimiter between prompts.
    """
    expanded: list[str] = []
    for t in post_do_tokens:
        s = str(t)
        while s.endswith(","):
            core = s[:-1].strip()
            if core:
                expanded.append(core)
            expanded.append(",")
            s = ""
        if s.strip():
            expanded.append(s.strip())
    groups: list[list[str]] = []
    cur: list[str] = []
    for t in expanded:
        if t == ",":
            groups.append(cur)
            cur = []
        else:
            cur.append(t)
    groups.append(cur)
    prompts: list[str] = []
    for g in groups:
        if not g:
            continue
        if len(g) != 1:
            raise ValueError(
                "each /while body prompt must be one quoted phrase; separate prompts with commas "
                '(example: /while "c" do "step one", "step two")'
            )
        prompts.append(g[0])
    if not prompts:
        raise ValueError("missing body prompts after do")
    return prompts


def parse_while_repl_tokens(toks: list[str]) -> Tuple[int, str, list[str]]:
    """
    Parse shlex tokens after splitting the full REPL line.
    Expected: ['/while', optional --max N, ...condition..., 'do', ...body tokens...]
    Body: one or more comma-separated quoted prompts (space before comma optional).
    """
    if not toks or toks[0].lower() != "/while":
        raise ValueError("internal: not a /while command")
    i = 1
    max_iter = 50
    if i + 1 < len(toks) and toks[i] == "--max":
        try:
            max_iter = int(toks[i + 1], 10)
        except (ValueError, TypeError) as e:
            raise ValueError("--max must be followed by a positive integer") from e
        if max_iter < 1:
            raise ValueError("--max must be at least 1")
        i += 2
    rest = toks[i:]
    if len(rest) < 3:
        raise ValueError("missing condition, 'do', or body")
    do_idx = None
    for j, t in enumerate(rest):
        if str(t).lower() == "do":
            do_idx = j
            break
    if do_idx is None:
        raise ValueError(
            "missing literal 'do' between condition and body "
            '(example: /while "tests still failing" do "fix failures and rerun")'
        )
    if do_idx == 0 or do_idx >= len(rest) - 1:
        raise ValueError("condition and body must be non-empty")
    condition = " ".join(rest[:do_idx]).strip()
    post_do = rest[do_idx + 1 :]
    if not post_do:
        raise ValueError("missing body after do")
    if not condition:
        raise ValueError("condition must be non-empty")
    body_prompts = post_do_tokens_to_body_prompts(post_do)
    return max_iter, condition, body_prompts


def call_while_condition_judge(
    condition: str,
    messages: list,
    *,
    primary_profile,
    verbose: int,
    default_primary_llm_profile: Callable[[], object],
    call_hosted_chat_plain: Callable,
    call_ollama_plaintext: Callable[[list, str], str],
    ollama_model: str,
    scalar_to_str_fn: Callable[..., str],
) -> int:
    excerpt = while_conversation_excerpt_for_judge(
        messages, scalar_to_str_fn=scalar_to_str_fn
    )
    user_body = (
        "Evaluate this /while CONDITION as TRUE or FALSE right now "
        "(reply 1 if TRUE — keep looping; 0 if FALSE — exit):\n"
        f"{condition}\n\n"
        "--- Conversation excerpt (may be truncated) ---\n"
        f"{excerpt}"
    )
    judge_msgs = [
        {"role": "system", "content": WHILE_JUDGE_SYSTEM},
        {"role": "user", "content": user_body},
    ]
    prof = primary_profile or default_primary_llm_profile()
    if prof.backend == "hosted":
        raw = call_hosted_chat_plain(judge_msgs, prof)
    else:
        raw = call_ollama_plaintext(judge_msgs, ollama_model)
    bit = parse_while_judge_bit(raw)
    if verbose >= 1:
        preview = (raw or "").replace("\n", " ").strip()
        if len(preview) > 200:
            preview = preview[:199] + "…"
        print(f"[/while judge] model={prof.backend!r} raw={preview!r} -> {bit}")
    return bit
