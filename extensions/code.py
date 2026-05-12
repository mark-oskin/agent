"""
Feature pipeline: designer → coder → reviewer ⟲ → tester ⟲.

Load with ``/load extensions/code.py`` (from repo root) or an absolute path.

**Multi-lane (``agent_tui.py``):** when ``python_fork_background_agent`` and ``python_delegate_line``
are both set, post-load forks background lanes (designer, coder, reviewer, tester) and each step
delegates to the matching lane.

**Single-lane (plain ``agent`` / CLI):** when fork is not wired, the same prompts run on the
current session via ``execute_line`` (one conversation; no ``/fork``).

Boot strings apply only to forked lanes. Per-step prompts include workspace (``session_cwd``),
rubric guidance, and a shared ``---PIPELINE---`` verdict block.

Pipeline loop limits are **defaults in this module**; optional overrides via
``/set extensions code_pipeline set <key> <value>`` (then ``/set save``) merge on top of those defaults.

Optional ``/code`` flags (any combination, before the description): ``--skip_design`` (use request as plan;
after a **tester** failure, run **design** to revise the plan), ``--skip_review``, ``--skip_test``.
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
    "design_review_max": 10,
    "code_test_max": 10,
    "inner_round_max": 5,
    "parse_fail_max": 10,
    "user_ask_max_len": 8000,
}

# Replies at least this long without a parseable verdict block get a follow-up nudge and do not
# increment the global ``parse_fails`` counter (``inner_round_max`` still limits tries).
_SUBSTANTIAL_REPLY_MIN_CHARS = 200


# Optional ``/code`` flags (leading tokens, whitespace-separated).
_CODE_FLAG_SKIP_DESIGN = "--skip_design"
_CODE_FLAG_SKIP_REVIEW = "--skip_review"
_CODE_FLAG_SKIP_TEST = "--skip_test"
_CODE_KNOWN_FLAGS = frozenset({_CODE_FLAG_SKIP_DESIGN, _CODE_FLAG_SKIP_REVIEW, _CODE_FLAG_SKIP_TEST})


class _CodePipelineFlags(NamedTuple):
    skip_design: bool = False
    skip_review: bool = False
    skip_test: bool = False


def _parse_code_rest(rest: str) -> tuple[_CodePipelineFlags, str]:
    """
    Parse leading ``--skip_*`` tokens; remainder is the user request text.

    Example: ``"--skip_design --skip_test fix login"`` → flags + ``"fix login"``.
    """
    parts = (rest or "").split()
    skip_design = False
    skip_review = False
    skip_test = False
    i = 0
    while i < len(parts) and parts[i] in _CODE_KNOWN_FLAGS:
        if parts[i] == _CODE_FLAG_SKIP_DESIGN:
            skip_design = True
        elif parts[i] == _CODE_FLAG_SKIP_REVIEW:
            skip_review = True
        elif parts[i] == _CODE_FLAG_SKIP_TEST:
            skip_test = True
        i += 1
    ask = " ".join(parts[i:]).strip()
    return _CodePipelineFlags(skip_design, skip_review, skip_test), ask


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
        "---PIPELINE---\nVERDICT: PASS or FAIL\nSUMMARY: …\n---END---\n\n"
        "You must respond with JSON only. Put that entire block inside the JSON string value of "
        '"answer" (literal newlines inside the string are fine). Example shape:\n'
        '{"action":"answer","answer":"…your prose…\\n---PIPELINE---\\nVERDICT: PASS\\nSUMMARY: …\\n---END---\\n"}\n'
    )


# Placed at the end of the orchestrator-composed user request (right before the runner adds
# "Respond with JSON only…") so it stays in the model's last-token context for this turn.
_TRAIL_PIPELINE_REMINDER = (
    "\n\nAUTOMATION — the orchestrator parses only your JSON \"answer\" string. "
    'That string MUST contain the literal lines ---PIPELINE---\nVERDICT: PASS or FAIL\nSUMMARY: …\n---END---\n'
    "exactly as in the template above. If you omit them, this lane will be retried and the pipeline stalls."
)


def _empty_prior_verdict_nudge() -> str:
    """When the lane returned nothing usable, still push a strict retry (``prev_reply`` was empty)."""
    return (
        "ORCHESTRATION — your last turn produced no usable assistant text for the orchestrator "
        "(empty answer, step limit, or parse failure before text was captured). Reply once with JSON only:\n"
        '{"action":"answer","answer":"…\\n---PIPELINE---\\nVERDICT: FAIL\\nSUMMARY: what went wrong\\n---END---\\n"}\n'
        "Use PASS only if you actually completed the work and included the block as required."
    )


def _missing_verdict_nudge(prior_reply: str) -> str:
    """Append to a follow-up user prompt when the lane's last answer did not contain a parseable verdict."""
    tail = (prior_reply or "").strip()
    if len(tail) > 1600:
        tail = tail[-1600:]
    return (
        "ORCHESTRATION — your previous reply in this lane could not be parsed: it must end with "
        "exactly one block in this form (include the literal ---END--- line). "
        "Put that block inside your JSON answer string value, not after the closing brace.\n"
        "---PIPELINE---\n"
        "VERDICT: PASS or FAIL\n"
        "SUMMARY: …\n"
        "---END---\n"
        "Put your status, file paths, and commands you ran in SUMMARY. If work is incomplete, use VERDICT: FAIL.\n"
        "Reply again now (you may be brief outside the block).\n"
        + (f"\n--- Excerpt of your prior reply (reference only) ---\n{tail}\n" if tail else "")
    )


def _multilane_pipeline_available(session) -> bool:
    """True when the host can fork background lanes and delegate to them (e.g. ``agent_tui``)."""
    return (
        getattr(session, "python_fork_background_agent", None) is not None
        and getattr(session, "python_delegate_line", None) is not None
    )


def _delegate(session, role: str, prompt: str) -> str:
    text = (prompt or "").strip()
    if _multilane_pipeline_available(session):
        return _text_from_delegate(session.python_delegate_line(role, text))
    el = getattr(session, "execute_line", None)
    if not callable(el):
        raise RuntimeError("single-lane pipeline requires session.execute_line")
    return _text_from_delegate(el(text))


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
        prev_reply = last_text
        # Prepend contract + trail reminder (runner appends "Respond with JSON only" after user_query).
        body = build_prompt(prev_reply).strip()
        prompt = (
            _verdict_reminder().strip()
            + "\n\n────────\n\n"
            + body
            + _TRAIL_PIPELINE_REMINDER
        )
        if i > 0:
            if prev_reply.strip():
                prompt = prompt + "\n\n" + _missing_verdict_nudge(prev_reply)
            else:
                prompt = prompt + "\n\n" + _empty_prior_verdict_nudge()
        last_text = _delegate(session, role, prompt)
        parsed = parse_pipeline_verdict(last_text)
        if parsed:
            verdict, summary = parsed
            one_line = " ".join(summary.split())
            prev = (one_line[:160] + "…") if len(one_line) > 160 else one_line
            _msg(f"← {role}: parsed VERDICT={'PASS' if verdict else 'FAIL'}; SUMMARY preview: {prev!r}")
            return parsed
        substantial = len((last_text or "").strip()) >= _SUBSTANTIAL_REPLY_MIN_CHARS
        if substantial:
            _msg(
                f"← {role}: no parseable ---PIPELINE--- block (substantial reply, {len((last_text or '').strip())} chars); "
                f"will retry with reminder (parse_fail cap unchanged: {parse_fails[0]}/{lim.parse_fail_max})."
            )
        else:
            parse_fails[0] += 1
            _msg(
                f"← {role}: no valid ---PIPELINE--- block (parse issues: {parse_fails[0]}/{lim.parse_fail_max}); "
                f"will retry if attempts remain."
            )
            if parse_fails[0] >= lim.parse_fail_max:
                break
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

The runner requires JSON only: put the ---PIPELINE--- through ---END--- block inside your final action answer string.

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

The runner requires JSON only: put the ---PIPELINE--- through ---END--- block inside your final action answer string (the long text value), not after the JSON object.

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

The runner requires JSON only: put the ---PIPELINE--- through ---END--- block inside your final action answer string.

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

The runner requires JSON only: put the ---PIPELINE--- through ---END--- block inside your final action answer string.

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
    multilane = _multilane_pipeline_available(session)
    if not multilane and not callable(getattr(session, "execute_line", None)):
        return (
            "[code] No pipeline host: need either (1) agent_tui with fork + delegate hooks, or "
            "(2) a normal session with execute_line for single-lane mode."
        )

    lim = _pipeline_limits(session)

    flags, raw_ask = _parse_code_rest(user_ask)
    ask = " ".join(raw_ask.split()).strip()
    if not ask:
        return (
            "[code] Usage: /code [--skip_design] [--skip_review] [--skip_test] "
            "<description of the feature or change>"
        )
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

    flag_note = []
    if flags.skip_design:
        flag_note.append("skip_design")
    if flags.skip_review:
        flag_note.append("skip_review")
    if flags.skip_test:
        flag_note.append("skip_test")
    active = f" [flags: {', '.join(flag_note)}]" if flag_note else ""

    if multilane:
        _msg(
            "Starting pipeline (multi-lane: each step delegates to that lane)."
            + active
            + " Flow: designer plan → coder ↔ reviewer⇄designer → tester; "
            f"tester FAIL re-enters coder (max {lim.code_test_max} test cycles)."
        )
    else:
        _msg(
            "Starting pipeline (single-lane: all steps on this session via execute_line)."
            + active
            + f" Max {lim.design_review_max} design/review rounds, {lim.code_test_max} test/code cycles."
        )

    if flags.skip_design:
        designer_summary = (
            "[Initial design phase skipped by --skip_design. "
            "Implement directly from the user request; treat this block as the working specification.]\n\n"
            f"USER REQUEST:\n{ask}\n"
        )
        _msg("Design skipped (--skip_design): using user request as the working plan for the coder.")
    else:

        def designer_prompt_initial(_: str) -> str:
            return (
                _workspace_blurb(session)
                + "The user wants this feature or change in the codebase.\n\n"
                f"USER REQUEST:\n{ask}\n\n"
                + "Discuss steps and testing as required. The pipeline contract is prepended to this message.\n"
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

            if flags.skip_review:
                reviewer_summary = "(Review skipped by user flag --skip_review.)"
                reviewer_pass = True
                _msg("Reviewer skipped (--skip_review) — proceeding.")
                break

            def reviewer_prompt(_: str) -> str:
                return (
                    _workspace_blurb(session)
                    + "You only have the text below — not a full git diff. Base judgment on consistency, "
                    "plausibility, and whether the coder's SUMMARY gives enough to trust the change.\n\n"
                    f"USER REQUEST:\n{ask}\n\n"
                    f"DESIGN:\n{designer_summary}\n\n"
                    f"IMPLEMENTATION (coder SUMMARY):\n{coder_summary}\n\n"
                    "If critical verification needs files you were not given, use VERDICT: FAIL and say what is missing.\n"
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

        if flags.skip_test:
            tester_summary = "(Tests skipped by user flag --skip_test.)"
            ok_t = True
            _msg("Tester skipped (--skip_test).")
        else:

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
            bits = ["[code] Pipeline complete."]
            if flags.skip_review:
                bits.append("Reviewer skipped (--skip_review).")
            if flags.skip_test:
                bits.append("Tester skipped (--skip_test).")
            if not flags.skip_review and not flags.skip_test:
                bits.append("Reviewer and tester PASS.")
            elif not flags.skip_test:
                bits.append("Tester PASS.")
            elif not flags.skip_review:
                bits.append("Reviewer PASS.")
            head = " ".join(bits) if len(bits) > 1 else bits[0]
            return (
                f"{head}\n\n"
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

        if flags.skip_design:
            _msg("Tester failed with --skip_design: running designer to revise the plan from test output.")

            def designer_after_test_fail(_: str) -> str:
                return (
                    _workspace_blurb(session)
                    + "Automated tests failed or reported problems. Produce an updated implementation plan "
                    "that addresses the failures (the coder will follow this plan on the next pass).\n\n"
                    f"USER REQUEST:\n{ask}\n\n"
                    f"PRIOR PLAN / SPEC:\n{designer_summary}\n\n"
                    f"TESTER OUTPUT:\n{tester_summary}\n\n"
                    "Use VERDICT: PASS with the revised plan in SUMMARY unless you cannot salvage a reasonable approach.\n"
                )

            ok_td, designer_summary = _parse_or_retry(
                session,
                "designer",
                designer_after_test_fail,
                parse_fails=parse_fails,
                stage="revise plan after tester FAIL (--skip_design mode)",
                lim=lim,
            )
            if not ok_td:
                return (
                    "[code] designer could not produce a plan after tester failure "
                    f"(skip_design recovery).\n{designer_summary[:1500]}"
                )
            tester_feedback = ""
            _msg("Designer updated plan after test failure; re-entering coder with new design.")
        else:
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
        "/code [--skip_design] [--skip_review] [--skip_test] <description> — "
        "designer → coder ↔ reviewer → tester pipeline."
    )
    registry.register_command("code", _cmd_code)
    if _multilane_pipeline_available(session):
        return list(_POST_LOAD_LINES)
    return []
