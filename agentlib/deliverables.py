"""Heuristics and follow-up copy for written-deliverable (document) tasks."""

from __future__ import annotations

import re
from typing import Callable


def user_wants_written_deliverable(user_query: str) -> bool:
    """Heuristic: user asked for a substantive written artifact (not just Q&A)."""
    q = (user_query or "").strip().lower()
    if not q:
        return False
    if re.search(r"\b(write|draft|compose)\b", q) and re.search(
        r"\b(letter|memo|e-?mail)\b", q
    ):
        return True
    if re.search(r"\b(document|essay|report|manuscript|white\s*paper|writeup|write-up)\b", q):
        return True
    if re.search(r"\b(page|pages)\b", q) and re.search(
        r"\b(write|draft|produce|author|compose|deliver)\b", q
    ):
        return True
    if re.search(r"\bwrite\s+the\s+document\b", q) or re.search(r"\bdon't\s+just\s+do\s+the\s+outline\b", q):
        return True
    return False


def deliverable_skip_mandatory_web(user_query: str) -> bool:
    """
    Do not inject router-mandated search_web for written deliverables unless the user asked for
    research, citations, or web-grounded facts. Otherwise models often mirror the whole prompt as
    a search query and loop on identical searches (extra JSON keys also used to bypass dedupe).
    """
    if not user_wants_written_deliverable(user_query):
        return False
    q = (user_query or "").strip().lower()
    if re.search(
        r"\b(sources|citations?|references?|bibliograph(y|ies)|research)\b|"
        r"\blook\s*up\b|\bverify\s+online\b|from\s+(the\s+)?web\b|"
        r"\bfrom\s+news\b|\bwikipedia\b|\binclude\s+urls?\b|"
        r"\bcurrent\s+events\b|\blatest\s+news\b|\baccording\s+to\s+the\b",
        q,
    ):
        return False
    return True


def deliverable_followup_block(path: str, scalar_to_str: Callable) -> str:
    p = scalar_to_str(path, "").strip()
    return (
        "Deliverable reminder: The user asked for a written document, not a short summary. "
        "If you already used write_file, you must finish the task by reading that file back with read_file "
        f'and then responding with {{"action":"answer","answer":"..."}} that includes the FULL document text '
        f"(or clearly states the file path and pastes the full contents). Do not stop after fetch_page with only a synopsis. "
        f'Next step: call read_file with parameters.path == "{p}".'
    )


def answer_missing_written_body(answer: str, file_chars: int) -> bool:
    """True if final answer omits most of the written file content."""
    a = (answer or "").strip()
    if file_chars <= 0:
        return False
    if len(a) < int(file_chars * 0.85):
        return True
    return False


def deliverable_first_answer_followup(user_query: str, proposed_answer: str) -> str:
    """
    When the user asked for a letter/document but the model answered with no tools, the generic
    web-vs-memory self-check invites meta-rationalizations instead of the requested prose.
    """
    uq = (user_query or "").strip()
    ans = (proposed_answer or "").strip()
    return (
        "You just responded with action=answer without using any tools, but the user asked for a "
        "written deliverable (letter, memo, email, document, etc.).\n\n"
        "User request:\n"
        f"{uq}\n\n"
        "Your last answer (not acceptable as the final reply):\n"
        f"{ans}\n\n"
        "You must satisfy the request itself: put the **full text** they asked for in the answer "
        "(salutation through closing/signature for a letter), as if they could send or publish it. "
        "Do **not** reply with commentary about whether web search is needed, timeliness, or "
        "\"timeless\" policy questions—produce the artifact.\n\n"
        "Alternatively you may use write_file with the complete text, then read_file that path, "
        "then action answer with the same full text.\n\n"
        "Respond with JSON only: "
        '{"action":"answer","answer":"..."} with the complete writing, or '
        '{"action":"tool_call","tool":"write_file",...} as appropriate.'
    )
