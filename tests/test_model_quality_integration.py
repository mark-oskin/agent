"""
Live Ollama model quality comparisons (optional).

These tests call your local Ollama HTTP API and compare objective rubric scores:
- Coding:   lfm2:latest  vs  qwen3-coder-next
- General:  lfm2:latest  vs  gemma4:26b

Enable with:
  RUN_OLLAMA_QUALITY=1 uv run pytest tests/test_model_quality_integration.py -v

Optional tuning:
  OLLAMA_HOST              (default http://localhost:11434)
  OLLAMA_MODEL_LFM         override lfm2 tag (default lfm2:latest)
  OLLAMA_MODEL_CODER       override coder ref (default qwen3-coder-next, resolves :latest)
  OLLAMA_MODEL_GENERAL     override general ref (default gemma4:26b)
  OLLAMA_QUALITY_TOLERANCE  max points lfm2 may trail the reference (default 18)
  OLLAMA_QUALITY_MIN_SCORE  minimum acceptable score for lfm2 (default 50)
  OLLAMA_CHAT_TIMEOUT_SEC   per-request timeout (default 180)
"""

from __future__ import annotations

import ast
import os
import re
from typing import List, Tuple

import pytest
import requests

pytestmark = pytest.mark.ollama_quality

MODEL_LFM = os.environ.get("OLLAMA_MODEL_LFM", "lfm2:latest")
MODEL_CODER = os.environ.get("OLLAMA_MODEL_CODER", "qwen3-coder-next")
MODEL_GENERAL = os.environ.get("OLLAMA_MODEL_GENERAL", "gemma4:26b")


def _enabled() -> bool:
    return os.environ.get("RUN_OLLAMA_QUALITY", "").strip() in ("1", "true", "yes", "on")


def _base() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _timeout() -> int:
    return int(os.environ.get("OLLAMA_CHAT_TIMEOUT_SEC", "180"))


def _tolerance() -> int:
    return int(os.environ.get("OLLAMA_QUALITY_TOLERANCE", "18"))


def _min_score() -> int:
    return int(os.environ.get("OLLAMA_QUALITY_MIN_SCORE", "50"))


def _ollama_list_models() -> List[str]:
    r = requests.get(f"{_base()}/api/tags", timeout=10)
    r.raise_for_status()
    data = r.json() or {}
    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]


def _resolve_model_tag(requested: str, names: set[str]) -> str | None:
    """Map user-facing names to Ollama tags (e.g. qwen3-coder-next -> qwen3-coder-next:latest)."""
    if requested in names:
        return requested
    if ":" not in requested and f"{requested}:latest" in names:
        return f"{requested}:latest"
    pref = [n for n in names if n == requested or n.startswith(requested + ":")]
    if len(pref) == 1:
        return pref[0]
    return None


def _chat_plain(model: str, system: str, user: str) -> str:
    r = requests.post(
        f"{_base()}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.25},
        },
        timeout=_timeout(),
    )
    r.raise_for_status()
    msg = (r.json() or {}).get("message") or {}
    return (msg.get("content") or "").strip()


def _extract_python_fenced(text: str) -> str:
    for pat in (
        r"```(?:python)?\s*\n(.*?)```",
        r"```(?:python)?\s+(.*?)```",
    ):
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def _score_coding_reverse_words(response: str) -> Tuple[int, str]:
    """
    Return (score 0-100, reason). Requires a fenced Python block defining reverse_words(str)->str.
    """
    code = _extract_python_fenced(response)
    if not code:
        return 0, "no_fenced_python"
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return 5, f"syntax_error:{e}"

    # Disallow obvious imports / subprocess in this tiny exercise
    banned = (r"\bimport\s+os\b", r"\bimport\s+subprocess\b", r"\b__import__\b", r"\beval\s*\(", r"\bexec\s*\(")
    low = code.lower()
    if any(re.search(p, low) for p in banned):
        return 10, "banned_construct"

    ns: dict = {}
    try:
        exec(compile(tree, "<model>", "exec"), {"__builtins__": __builtins__}, ns)
    except Exception as e:  # noqa: BLE001
        return 15, f"exec_error:{e}"

    fn = ns.get("reverse_words")
    if not callable(fn):
        return 20, "no_reverse_words_callable"

    score = 40
    reasons: List[str] = ["parsed+loaded"]

    tests = [
        ("a b c", "c b a"),
        ("  hello   world  ", "world hello"),
        ("single", "single"),
        ("", ""),
    ]
    for inp, expected in tests:
        try:
            got = fn(inp)
        except Exception as e:  # noqa: BLE001
            return 25, f"runtime_error:{e}"
        if got != expected:
            return 30, f"wrong_output:{inp!r}->{got!r}!={expected!r}"
    score += 50
    reasons.append("all_unit_checks")

    if len(code) > 800:
        score = max(0, score - 5)
        reasons.append("verbose_penalty")

    return min(100, score), "+".join(reasons)


def _score_general_psk31(response: str) -> Tuple[int, str]:
    """Rubric for a short explanatory answer (not looking for one verbatim answer)."""
    t = (response or "").strip()
    if len(t) < 120:
        return 0, "too_short"
    low = t.lower()
    score = 0
    bits: List[str] = []

    if "psk31" in low or "psk-31" in low:
        score += 25
        bits.append("mentions_psk31")
    if re.search(r"\b(bpsk|phase|psk|psk-?31)\b", low) or "psk31" in low:
        score += 20
        bits.append("modulation_terms")
    if re.search(r"\b(31|baud|bandwidth|narrow|hz)\b", low):
        score += 15
        bits.append("speed_or_bw")
    if re.search(r"\b(amateur|ham|hf|radio|keyboard)\b", low):
        score += 20
        bits.append("context")
    # Coherence / not a refusal
    if "can't" in low or "cannot" in low or "i'm unable" in low:
        score = max(0, score - 25)
        bits.append("refusal_penalty")

    if len(t) > 3500:
        score = max(0, score - 10)
        bits.append("too_long_penalty")

    return min(100, score), "+".join(bits) if bits else "weak"


@pytest.fixture(scope="module")
def ollama_models():
    if not _enabled():
        pytest.skip("Set RUN_OLLAMA_QUALITY=1 to run live Ollama model quality comparisons.")
    try:
        names = set(_ollama_list_models())
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Ollama not reachable at {_base()!r}: {e}")

    out = {}
    for key, want in (("lfm", MODEL_LFM), ("coder", MODEL_CODER), ("general", MODEL_GENERAL)):
        tag = _resolve_model_tag(want, names)
        if not tag:
            pytest.skip(f"No Ollama model matching {want!r} (have {sorted(names)})")
        out[key] = tag
    return out


def test_coding_lfm2_not_much_worse_than_qwen3_coder(ollama_models):
    system = "You are a careful Python programmer. Follow instructions exactly."
    user = (
        "Write exactly one markdown fenced code block labeled python.\n"
        "Inside it, define only this function (no input(), no main, no tests, no explanation):\n\n"
        "def reverse_words(s: str) -> str:\n"
        "    '''Return words of s in reverse order; words are split on whitespace; "
        "preserve single spaces between words in the output.'''\n"
    )

    r_lfm = _chat_plain(ollama_models["lfm"], system, user)
    r_qwen = _chat_plain(ollama_models["coder"], system, user)

    s_lfm, why_lfm = _score_coding_reverse_words(r_lfm)
    s_qwen, why_qwen = _score_coding_reverse_words(r_qwen)

    assert s_lfm >= _min_score(), (
        f"lfm2 coding score too low: {s_lfm} ({why_lfm}). Raw (truncated): {r_lfm[:500]!r}"
    )
    assert s_lfm + _tolerance() >= s_qwen, (
        f"lfm2 ({s_lfm},{why_lfm}) trails qwen3-coder-next ({s_qwen},{why_qwen}) "
        f"by more than tolerance {_tolerance()}"
    )


def test_general_lfm2_not_much_worse_than_gemma(ollama_models):
    system = "You are a helpful assistant. Answer clearly and concretely."
    user = (
        "In 4–8 sentences for a technically curious reader, explain what PSK31 is in amateur radio, "
        "why it became popular, and one limitation. Do not use bullet lists; write as a short paragraph."
    )

    r_lfm = _chat_plain(ollama_models["lfm"], system, user)
    r_gem = _chat_plain(ollama_models["general"], system, user)

    s_lfm, why_lfm = _score_general_psk31(r_lfm)
    s_gem, why_gem = _score_general_psk31(r_gem)

    assert s_lfm >= _min_score(), (
        f"lfm2 general score too low: {s_lfm} ({why_lfm}). Raw (truncated): {r_lfm[:800]!r}"
    )
    assert s_lfm + _tolerance() >= s_gem, (
        f"lfm2 ({s_lfm},{why_lfm}) trails gemma4:26b ({s_gem},{why_gem}) "
        f"by more than tolerance {_tolerance()}"
    )


def test_reference_models_answer_nonzero_scores(ollama_models):
    """Sanity: reference models should clear the rubrics on a good day (helps debug env issues)."""
    system_c = "You are a careful Python programmer. Follow instructions exactly."
    user_c = (
        "Write exactly one markdown fenced code block labeled python.\n"
        "Inside it, define only this function:\n\n"
        "def reverse_words(s: str) -> str:\n"
        "    '''Return words of s in reverse order; words split on whitespace.'''\n"
    )
    s_qwen, _ = _score_coding_reverse_words(_chat_plain(ollama_models["coder"], system_c, user_c))
    assert s_qwen >= 70, f"qwen3-coder-next unexpectedly weak on coding rubric: {s_qwen}"

    system_g = "You are a helpful assistant."
    user_g = (
        "In 4–8 sentences, explain what PSK31 is in amateur radio and why operators use it. "
        "One short paragraph, no bullet list."
    )
    s_gem, _ = _score_general_psk31(_chat_plain(ollama_models["general"], system_g, user_g))
    assert s_gem >= 60, f"gemma4:26b unexpectedly weak on general rubric: {s_gem}"
