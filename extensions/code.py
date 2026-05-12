"""
TUI extension: multi-lane feature pipeline (designer → coder → reviewer ⟲ → tester ⟲).

Load with ``/load extensions/code.py`` (from repo root) or an absolute path.

Requires ``agent_tui.py`` (``python_delegate_line`` + ``python_fork_background_agent``).

Lane labels (case-insensitive): designer, coder, reviewer, tester — created via post-load
``/fork_background`` if they do not already exist.

Boot and per-step prompts include workspace (``session_cwd``), scope/rubric guidance, and a shared
``---PIPELINE---`` verdict block; ``SUMMARY`` may span multiple lines (newlines preserved when parsed).

Pipeline loop limits are **defaults in this module**; optional overrides via
``/set extensions code_pipeline set <key> <value>`` (then ``/set save``) merge on top of those defaults.
"""

from __future__ import annotations

import re
import threading
from typing import NamedTuple, Optional, Tuple

from agentlib.repl_extensions import ReplExtensionRegistry
from agentlib.session import SessionLineResult
from agentlib.sink import sink_print_compat
from agentlib.tui_parse import format_fork_command_line

# Prefs id under ``session.settings.extensions`` (optional overrides via ``/set extensions``).
EXTENSION_SETTINGS_ID = "code_pipeline"

# Self-contained pipeline tunables (override per-session with ``/set extensions code_pipeline …``).
_PIPELINE_DEFAULTS: dict[str, int] = {
    "design_review_max": 5,
    "code_test_max": 5,
    "inner_round_max": 3,
    "parse_fail_max": 10,
    "user_ask_max_len": 8000,
}


class _PipelineLimits(NamedTuple):
    design_review_max: int
    code_test_max: int
    inner_round_max: int
    parse_fail_max: int
    user_ask_max_len: int


def _pipeline_limits(session) -> _PipelineLimits:
    base = dict(_PIPELINE_DEFAULTS)
    st = getattr(session, "settings", None)
    if st is not None:
        raw = st.get(("extensions", EXTENSION_SETTINGS_ID))
        if isinstance(raw, dict):
            for k in _PIPELINE_DEFAULTS:
                if k in raw:
                    base[k] = raw[k]

    def _int(name: str, *, minimum: int = 1) -> int:
        v = base.get(name)
        try:
            n = int(str(v).strip(), 10)
        except Exception:
            n = int(_PIPELINE_DEFAULTS[name])
        return max(minimum, n)

    return _PipelineLimits(
        design_review_max=_int("design_review_max"),
        code_test_max=_int("code_test_max"),
        inner_round_max=_int("inner_round_max"),
        parse_fail_max=_int("parse_fail_max"),
        user_ask_max_len=_int("user_ask_max_len", minimum=256),
    )

# --- Verdict block ------------------------------------------------------------

_PIPELINE_BLOCK = re.compile(
    r"---PIPELINE---\s*"
    r"VERDICT:\s*(PASS|FAIL)\s*"
    r"SUMMARY:\s*(.*?)\s*"
    r"---END---",
    re.IGNORECASE | re.DOTALL,
)


def parse_pipeline_verdict(text: str) -> Optional[Tuple[bool, str]]:
    """
    Extract the last ``---PIPELINE--- … ---END---`` block.

    Returns ``(is_pass, summary)`` or ``None`` if missing / malformed.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    matches = list(_PIPELINE_BLOCK.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    verdict = m.group(1).upper()
    raw = (m.group(2) or "").strip()
    if not raw:
        return None
    # Preserve newlines (bullets); trim each line's internal whitespace only.
    lines = [" ".join(line.split()) for line in raw.splitlines()]
    summary = "\n".join(L for L in lines if L).strip()
    if not summary:
        return None
    return verdict == "PASS", summary


def _text_from_delegate(res: object) -> str:
    if not isinstance(res, dict):
        return str(res)
    if res.get("type") == "turn":
        return str(res.get("answer") or "")
    if res.get("type") == "command":
        return str(res.get("output") or "")
    return str(res)


def _workspace_blurb(session) -> str:
    cwd = (getattr(session, "session_cwd", None) or "").strip()
    if not cwd:
        cwd = "(unset — use the process working directory)"
    return (
        "Workspace:\n"
        f"- Use this session's working directory for shells, tests, and relative paths: {cwd}\n"
        "- Run `/cd` on the repo root first if you are not already there.\n"
    )


def _verdict_reminder() -> str:
    return (
        "Always end with exactly one block of this form (no extra text after ---END--- is required, "
        "but the block must be parseable):\n"
        "---PIPELINE---\nVERDICT: PASS or FAIL\nSUMMARY: …\n---END---\n"
    )


def _delegate(session, role: str, prompt: str) -> str:
    dl = session.python_delegate_line
    if dl is None:
        raise RuntimeError("python_delegate_line is not configured")
    return _text_from_delegate(dl(role, prompt.strip()))


def _msg(line: str) -> None:
    """Progress line on the orchestrator lane (stdout / emit sink when active)."""
    sink_print_compat(f"[code] {line}")


def _parse_or_retry(
    session,
    role: str,
    build_prompt,
    *,
    parse_fails: list,
    stage: str,
    lim: _PipelineLimits,
) -> Tuple[bool, str]:
    """Run up to ``lim.inner_round_max`` delegate rounds until a verdict parses."""
    last_text = ""
    for i in range(lim.inner_round_max):
        attempt = f"{i + 1}/{lim.inner_round_max}" if lim.inner_round_max > 1 else "1/1"
        _msg(f"→ {role}: {stage} (delegate {attempt}; parse issues so far: {parse_fails[0]}) …")
        last_text = _delegate(session, role, build_prompt(last_text))
        parsed = parse_pipeline_verdict(last_text)
        if parsed:
            verdict, summary = parsed
            one_line = " ".join(summary.split())
            prev = (one_line[:160] + "…") if len(one_line) > 160 else one_line
            _msg(f"← {role}: parsed VERDICT={'PASS' if verdict else 'FAIL'}; SUMMARY preview: {prev!r}")
            return parsed
        parse_fails[0] += 1
        _msg(
            f"← {role}: no valid ---PIPELINE--- block (parse issues: {parse_fails[0]}/{lim.parse_fail_max}); "
            f"will retry if attempts remain."
        )
        if parse_fails[0] >= lim.parse_fail_max:
            break
    return False, (last_text or "")[:2000]


# --- Boot strings (no double quotes) -----------------------------------------

_BOOT_DESIGNER = """You are the designer agent for an automated code pipeline.

When you receive a user request, produce a concise implementation plan for this repository:
- concrete steps the coder should follow (ordering matters);
- how tests should validate the change (which suites, new vs updated tests);
- risks, edge cases, and assumptions where requirements are ambiguous (the orchestrator cannot answer follow-ups).

Make sure you study the directory and subdirectories from it that you are in to understand the current source code.
The directory may be empty if you are starting a new project.

End your message with EXACTLY one block in this form (SUMMARY may use multiple short lines).
Your response must include the highly stylized information below.  It will be parsed by code so do not forget any part of it (including the ---END---)


---PIPELINE---
VERDICT: PASS
SUMMARY: <your plan>
---END---

Use VERDICT: FAIL only if you cannot produce any reasonable plan at all (e.g. request is empty or contradictory with no safe default)."""

_BOOT_CODER = """You are the coder agent for an automated pipeline.

Scope: make the smallest change that satisfies the designer's plan and the user request.
Do not refactor unrelated code, rename public APIs, or reformat large areas unless the plan requires it.
Match existing style and patterns in the repo.

Implement using your tools from the session working directory. Add or update tests as the plan describes.

When the implementation is ready for review (code written, tests added/updated as planned), end with a block below.
Your response must include the highly stylized information below.  It will be parsed by code so do not forget any part of it (including the ---END---)

---PIPELINE---
VERDICT: PASS
SUMMARY: <multi-line OK: list paths touched, commands you ran (e.g. pytest -q), remaining risks>
---END---

Use VERDICT: FAIL if you are blocked (permissions, missing deps, contradictory requirements) after a genuine attempt."""

_BOOT_REVIEWER = """You are the reviewer agent.

You only see text summaries (user ask, design, implementation) — not the full repo diff.
Judge consistency with the user ask, whether the implementation summary plausibly matches the design,
and whether obvious gaps are acknowledged (tests, error handling, backwards compatibility).

You will need to study the directory and subdirectories to understand the repository.

Use this checklist mentally; if any item is clearly violated, prefer VERDICT: FAIL and say which item:
- Scope creep vs user ask / design
- Missing or contradictory tests vs what the design promised
- Break any explicitly documented public API

If you cannot verify something critical without seeing files, FAIL with SUMMARY explaining what you need (e.g. diff or paths).
Your response must include the highly stylized information below.  It will be parsed by code so do not forget any part of it (including the ---END---)

---PIPELINE---
VERDICT: PASS
SUMMARY: <brief review; note residual risks even on PASS>
---END---

Use VERDICT: FAIL if the implementation should be reworked."""

_BOOT_TESTER = """You are the tester agent.

Run automated tests from the session working directory (e.g. pytest, make test, npm test) using your tools.

You will need to scan the repo to find where the tests are located.

If the repo has no automated tests or you cannot find a standard command after a quick check, say so in SUMMARY
and use VERDICT: FAIL unless you performed and described an explicit alternative verification (e.g. ran a minimal repro script).
Your response must include the highly stylized information below.  It will be parsed by code so do not forget any part of it (including the ---END---)

---PIPELINE---
VERDICT: PASS
SUMMARY: <commands run, pass/fail, notable output>
---END---

Use VERDICT: FAIL if tests fail, were not run when they reasonably could be, or you could not verify meaningfully."""


def _fork_line(role: str, boot: str) -> str:
    """
    Build a ``/fork_background`` line using :func:`agentlib.tui_parse.format_fork_command_line`.

    Do not use :func:`shlex.quote` for ``boot``: apostrophes become ``'\"'\"'`` fragments containing
    ``"``, and :func:`agentlib.tui_parse.parse_fork_command` locates the payload by the first ``"``
    in the remainder, so parsing fails and the fork is skipped.
    """
    line = format_fork_command_line(role, [boot])
    if not line.startswith("/fork "):
        raise RuntimeError("format_fork_command_line returned unexpected prefix")
    return "/fork_background" + line[len("/fork") :]


_POST_LOAD_LINES = (
    _fork_line("designer", _BOOT_DESIGNER),
    _fork_line("coder", _BOOT_CODER),
    _fork_line("reviewer", _BOOT_REVIEWER),
    _fork_line("tester", _BOOT_TESTER),
)

_pipeline_lock = threading.Lock()


def _run_pipeline(session, user_ask: str) -> str:
    if session.python_fork_background_agent is None:
        return "[code] python_fork_background_agent is not set. Run under agent_tui.py."
    if session.python_delegate_line is None:
        return "[code] python_delegate_line is not set. Run under agent_tui.py."

    lim = _pipeline_limits(session)

    ask = " ".join(user_ask.split()).strip()
    if not ask:
        return "[code] Usage: /code <description of the feature or change>"
    if len(ask) > lim.user_ask_max_len:
        ask = ask[: lim.user_ask_max_len] + "\n…(truncated)"

    parse_fails = [0]
    design_review_cycles = 0
    code_test_cycles = 0

    designer_summary = ""
    coder_summary = ""
    reviewer_summary = ""
    tester_summary = ""
    tester_feedback = ""

    _msg(
        "Starting pipeline (blocking: each line waits for that lane to finish). "
        f"Flow: designer plan → coder ↔ reviewer⇄designer (max {lim.design_review_max} review rounds) → tester; "
        f"tester FAIL re-enters coder (max {lim.code_test_max} test cycles)."
    )

    def designer_prompt_initial(_: str) -> str:
        return (
            _workspace_blurb(session)
            + "The user wants this feature or change in the codebase.\n\n"
            f"USER REQUEST:\n{ask}\n\n"
            + "Discuss steps and testing as required. If you did not see the boot message, still follow this contract:\n"
            + _verdict_reminder()
        )

    ok, designer_summary = _parse_or_retry(
        session,
        "designer",
        designer_prompt_initial,
        parse_fails=parse_fails,
        stage="initial plan from user request",
        lim=lim,
    )
    if not ok:
        return f"[code] designer could not produce a passing plan (or parse failed). Last excerpt:\n{designer_summary[:1500]}"

    _msg("Initial designer plan accepted (VERDICT: PASS). Entering implement/review loop.")

    while code_test_cycles < lim.code_test_max:
        reviewer_pass = False
        _msg(
            f"— Test cycle {code_test_cycles + 1}/{lim.code_test_max} "
            f"(tester failures so far: {code_test_cycles}; design/review rounds used: {design_review_cycles})."
        )

        while design_review_cycles < lim.design_review_max:

            def coder_prompt(_: str) -> str:
                chunks = [
                    _workspace_blurb(session),
                    "The user asked for the following. The designer produced a plan.\n\n",
                    f"USER REQUEST:\n{ask}\n\n",
                    f"DESIGNER PLAN:\n{designer_summary}\n\n",
                ]
                if tester_feedback.strip():
                    chunks.append(
                        "PREVIOUS TEST / AUTOMATION FEEDBACK (address this):\n"
                        f"{tester_feedback}\n\n"
                    )
                chunks.append(
                    "Implement the change: minimal scope, match repo style, add/update tests per the plan.\n"
                    "VERDICT: PASS means the work is ready for review (not 'merged to main' unless you actually did that).\n"
                    "SUMMARY must list paths touched, commands you ran (e.g. pytest), and any remaining risks.\n"
                    + _verdict_reminder()
                )
                return "".join(chunks)

            c_stage = "implement from design"
            if tester_feedback.strip():
                c_stage += " (incorporating tester feedback)"
            ok_c, coder_summary = _parse_or_retry(
                session, "coder", coder_prompt, parse_fails=parse_fails, stage=c_stage, lim=lim
            )
            if not ok_c:
                return f"[code] coder blocked or failed verdict.\n{coder_summary[:1500]}"

            _msg(f"Coder step complete (design/review round index {design_review_cycles}).")

            def reviewer_prompt(_: str) -> str:
                return (
                    _workspace_blurb(session)
                    + "You only have the text below — not a full git diff. Base judgment on consistency, "
                    "plausibility, and whether the coder's SUMMARY gives enough to trust the change.\n\n"
                    f"USER REQUEST:\n{ask}\n\n"
                    f"DESIGN:\n{designer_summary}\n\n"
                    f"IMPLEMENTATION (coder SUMMARY):\n{coder_summary}\n\n"
                    "If critical verification needs files you were not given, use VERDICT: FAIL and say what is missing.\n"
                    + _verdict_reminder()
                )

            ok_r, reviewer_summary = _parse_or_retry(
                session,
                "reviewer",
                reviewer_prompt,
                parse_fails=parse_fails,
                stage="review implementation vs design",
                lim=lim,
            )
            if ok_r:
                reviewer_pass = True
                _msg("Reviewer PASS — proceeding to tester.")
                break

            design_review_cycles += 1
            _msg(
                f"Reviewer FAIL — asking designer to revise (review round {design_review_cycles}/{lim.design_review_max})."
            )
            if design_review_cycles >= lim.design_review_max:
                return (
                    f"[code] reviewer/design loop exceeded {lim.design_review_max} cycles.\n"
                    f"Last review summary:\n{reviewer_summary[:1200]}"
                )

            def redesign_prompt(_: str) -> str:
                return (
                    _workspace_blurb(session)
                    + "The reviewer rejected the implementation or design fit. Produce an updated plan.\n\n"
                    f"USER REQUEST:\n{ask}\n\n"
                    f"PREVIOUS DESIGN:\n{designer_summary}\n\n"
                    f"REVIEWER FEEDBACK:\n{reviewer_summary}\n\n"
                    "Use VERDICT: FAIL only if you still cannot produce any reasonable plan. "
                    "Otherwise VERDICT: PASS with the revised plan in SUMMARY.\n"
                    + _verdict_reminder()
                )

            ok_d2, designer_summary = _parse_or_retry(
                session,
                "designer",
                redesign_prompt,
                parse_fails=parse_fails,
                stage="revise plan after reviewer feedback",
                lim=lim,
            )
            if not ok_d2:
                return f"[code] designer could not recover after review FAIL.\n{designer_summary[:1500]}"
            _msg("Designer revised plan; re-running coder with updated design.")

        if not reviewer_pass:
            return "[code] internal: expected reviewer pass before tester (please report)."

        def tester_prompt(_: str) -> str:
            return (
                _workspace_blurb(session)
                + "The user requested a codebase change. Summaries follow.\n\n"
                f"USER REQUEST:\n{ask}\n\n"
                f"DESIGN:\n{designer_summary}\n\n"
                f"IMPLEMENTATION:\n{coder_summary}\n\n"
                f"REVIEW:\n{reviewer_summary}\n\n"
                "Run automated tests when possible. If none exist or no command is discoverable, say so clearly; "
                "use VERDICT: FAIL unless you document an explicit alternative check you performed.\n"
                + _verdict_reminder()
            )

        ok_t, tester_summary = _parse_or_retry(
            session,
            "tester",
            tester_prompt,
            parse_fails=parse_fails,
            stage="run tests / report",
            lim=lim,
        )
        if ok_t:
            design_snip = designer_summary if len(designer_summary) <= 500 else designer_summary[:500] + "…"
            return (
                "[code] Pipeline complete — reviewer and tester PASS.\n\n"
                f"Tester summary:\n{tester_summary}\n\n"
                f"Design (excerpt):\n{design_snip}"
            )

        code_test_cycles += 1
        _msg(
            f"Tester FAIL — scheduling another coder pass (tester failures now: {code_test_cycles}/{lim.code_test_max})."
        )
        if code_test_cycles >= lim.code_test_max:
            return (
                f"[code] test/code loop exceeded {lim.code_test_max} cycles.\n"
                f"Last tester output:\n{tester_summary[:1500]}"
            )
        tester_feedback = tester_summary
        design_review_cycles = 0

    return f"[code] aborted after {lim.code_test_max} test/code cycles (outer guard)."


def _cmd_code(session, rest: str) -> SessionLineResult:
    if not _pipeline_lock.acquire(blocking=False):
        return SessionLineResult(output="[code] Pipeline already running; wait for it to finish.")
    try:
        msg = _run_pipeline(session, rest)
        return SessionLineResult(output=msg)
    except Exception as e:
        return SessionLineResult(output=f"[code] error: {type(e).__name__}: {e}")
    finally:
        _pipeline_lock.release()


def register_repl(session, registry: ReplExtensionRegistry):
    registry.register_help(
        "/code <description> — run the multi-lane pipeline (designer → coder ↔ reviewer → tester) "
        "via /fork_background lanes. Tunables live in this module; optional prefs: "
        "/set extensions code_pipeline set <key> <value> (see module _PIPELINE_DEFAULTS keys)."
    )
    registry.register_command("code", _cmd_code)
    return list(_POST_LOAD_LINES)
