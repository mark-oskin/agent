"""Validate model-emitted Python for call_python and detect garbled answer text."""

from __future__ import annotations

import re
from typing import Optional

_PROSE_IN_CODE = re.compile(
    r"\b(let me|you need to|I (?:will|should|accidentally)|rewrite (?:this|it)|properly)\b",
    re.IGNORECASE,
)


def call_python_code_rejected_reason(code: str) -> Optional[str]:
    """Return a short reason when ``code`` must not be executed; None if compile succeeds."""
    if code is None:
        return "empty code string"
    text = code if isinstance(code, str) else str(code)
    if not text.strip():
        return "empty code string"
    if _PROSE_IN_CODE.search(text):
        return (
            "code contains natural-language prose mixed with Python "
            "(use action answer to show a script as text, or emit only valid Python in parameters.code)"
        )
    try:
        compile(text, "<call_python>", "exec")
    except SyntaxError as e:
        return f"{e.msg} at line {e.lineno}"
    except Exception as e:
        return str(e)
    return None


CALL_PYTHON_SHOWN_AS_TEXT_NOTE = (
    "The model tried to run call_python with invalid or incomplete Python "
    "(syntax error or explanations mixed into the code). "
    "Showing the script as text instead of executing it."
)


def prefer_display_answer(parsed_answer: str, raw_response: str) -> str:
    """Pick the best user-visible text when JSON parsing yielded a weak code fragment."""
    code = (parsed_answer or "").strip()
    raw = (raw_response or "").strip()
    if not raw:
        return code

    def _fence_blocks(text: str) -> list[str]:
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if not blocks:
            blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
        return [b.strip() for b in blocks if b.strip()]

    blocks = _fence_blocks(raw)
    if blocks:
        best = max(blocks, key=len)
        if len(best) >= max(len(code), 40):
            intro_end = raw.find("```")
            intro = raw[:intro_end].strip() if intro_end > 0 else ""
            if intro and len(intro) < 2000 and not intro.lstrip().startswith("{"):
                return f"{intro}\n\n```python\n{best}\n```"
            return best

    json_start = raw.find('{"action"')
    if json_start < 0:
        json_start = raw.find("{\n")
    if json_start > 80:
        prefix = raw[:json_start].strip()
        if prefix and not prefix.lstrip().startswith("{"):
            return prefix

    if answer_looks_like_garbled_tool_output(code):
        return raw if len(raw) > len(code) else code
    if "\\n" in code and code.count("\n") < 3 and len(raw) > len(code):
        return raw
    return code or raw


def answer_looks_like_garbled_tool_output(answer: str) -> bool:
    """True when an answer field is a broken JSON/code fragment, not user-facing text."""
    a = (answer or "").strip()
    if not a:
        return False
    # Literal \\n escapes with no real newlines — truncated JSON string body.
    if "\\n" in a and "\n" not in a and any(tok in a for tok in ("except", "__main__", "def ", "import ")):
        return True
    if re.match(r"^\{[^{}]*['\"]?\)", a):
        return True
    if a.endswith('\\"\\n}') or a.endswith('"\\n}') or a.endswith("\\n}"):
        return True
    if re.search(r"\\n\s+(except|if __name__|def |import )", a):
        return True
    return False
