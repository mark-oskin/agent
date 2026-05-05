#!/usr/bin/env python3
"""
foreach_line — batch prompts through the agent from the REPL
============================================================

Run **inside the agent** with ``/call_python`` so the injected ``ai()`` helper exists.
Each non-empty line from a file (or stdin) is sent as one REPL turn, same as typing it.

Example::

    /cd /path/to/project
    /call_python foreach_line.py -f classes.txt -t 'Summarize: {line}'

Shell-style redirection is approximated (the REPL is not a shell); ``< FILE`` is rewritten to ``-f FILE``::

    /call_python foreach_line.py -t 'Topic: {line}' < classes.txt

Options
-------

``-f`` / ``--file``     Input file (default: stdin; stdin is awkward in the REPL — prefer ``-f``).

``-t`` / ``--template``  Wrap each line; must contain ``{line}``.

``-c`` / ``--clear-history``  Before **each** prompt, clear in-memory chat history (same effect as ``/clear``). Default: off.

``--skip-comments``   Skip lines whose first non-whitespace character is ``#``.

``ai(line)`` runs ``AgentSession.execute_line`` (LLM turns and ``/`` commands).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys


def print_help() -> None:
    """Usage text when the script is invoked with no CLI arguments."""
    print(
        """foreach_line.py — send each input line to the agent as a prompt (via /call_python).

Run from the agent REPL (required — defines ai()):

  /cd <project-dir>                                    # so relative -f paths resolve
  /call_python path/to/foreach_line.py -f items.txt
  /call_python path/to/foreach_line.py -f items.txt -t 'Task for {line}: ...'
  /call_python path/to/foreach_line.py -f items.txt -c   # clear history before each line

Options:
  -f, --file PATH       Input file (one prompt per line). Recommended in REPL.
  -t, --template STR    Wrap each line; must include {line}
  -c, --clear-history   Clear chat history before each line (default: off)
  --skip-comments       Ignore lines starting with # (after whitespace)

Without -f, stdin is read (mainly useful if something pipes into the agent process).

See the module docstring at the top of foreach_line.py for more detail.
"""
    )


def _clear_session_history() -> None:
    """Match AgentSession /clear: drop messages and a few REPL scratch fields."""
    try:
        sess = session  # type: ignore[name-defined]
    except NameError:
        return
    sess.messages.clear()
    sess.last_reuse_skill_id = None
    sess.repl_last_user_query = None
    sess.repl_last_assistant_answer = None


def _resolve_input_file_path(p: str | None) -> str | None:
    """Resolve a user-supplied path using the agent session cwd when run via ``/call_python``."""
    if not p:
        return None
    p = os.path.expanduser(p.strip())
    if os.path.isabs(p):
        return os.path.normpath(p)
    try:
        sess = session  # type: ignore[name-defined]
        base = getattr(sess, "session_cwd", None)
        if base and str(base).strip():
            root = os.path.abspath(os.path.expanduser(str(base).strip()))
            return os.path.normpath(os.path.join(root, p))
    except NameError:
        pass
    return os.path.abspath(p)


def _normalize_argv_redirect(argv: list[str]) -> list[str]:
    """Turn ``script ... < path`` into ``script ... -f path`` (REPL has no real shell stdin redirect)."""
    if len(argv) <= 1:
        return argv
    out = [argv[0]]
    i = 1
    while i < len(argv):
        if argv[i] == "<" and i + 1 < len(argv):
            out.extend(["-f", argv[i + 1]])
            i += 2
        else:
            out.append(argv[i])
            i += 1
    return out


def _format_prompt(template: str | None, line: str) -> str:
    s = line.rstrip("\n\r")
    if template is None:
        return s
    if "{line}" not in template:
        raise SystemExit("error: --template must contain the placeholder {line}")
    return template.format(line=s)


def _emit_result(r: object) -> None:
    if not isinstance(r, dict):
        print(repr(r))
        return
    t = r.get("type")
    if t == "turn":
        ans = r.get("answer")
        if ans is not None and str(ans).strip():
            print(str(ans).rstrip())
        else:
            print(f"(turn finished; answered={r.get('answered')!r})")
    elif t == "command":
        out = r.get("output")
        if out:
            print(str(out).rstrip())
    else:
        print(r)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = _normalize_argv_redirect(sys.argv[:])

    if len(argv) <= 1:
        print_help()
        return 0

    p = argparse.ArgumentParser(
        description="Foreach input line → ai(prompt). Run via /call_python.",
        add_help=True,
    )
    p.add_argument("-f", "--file", default=None, help="Input file (default: stdin)")
    p.add_argument(
        "-t",
        "--template",
        default=None,
        help="Optional wrapper with {line}, e.g. 'Translate to French: {line}'",
    )
    p.add_argument(
        "-c",
        "--clear-history",
        action="store_true",
        help="Clear in-memory chat history before each prompt (like /clear). Default: off.",
    )
    p.add_argument(
        "--skip-comments",
        action="store_true",
        help="Skip lines that start with # after stripping leading whitespace",
    )
    args = p.parse_args(argv[1:])

    try:
        invoke = ai  # type: ignore[name-defined]
    except NameError:
        print(
            "foreach_line.py: must be run via /call_python so `ai` is defined "
            "(see agent session helpers).",
            file=sys.stderr,
        )
        return 2

    input_path = _resolve_input_file_path(args.file)
    fh_ctx = open(input_path, encoding="utf-8") if input_path else contextlib.nullcontext(sys.stdin)
    try:
        with fh_ctx as fh:
            for raw in fh:
                if args.skip_comments:
                    stripped = raw.lstrip()
                    if stripped.startswith("#"):
                        continue
                prompt = _format_prompt(args.template, raw)
                if not prompt.strip():
                    continue
                if args.clear_history:
                    _clear_session_history()
                result = invoke(prompt)
                _emit_result(result)
                if isinstance(result, dict) and result.get("quit"):
                    break
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
        return 0
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ in ("__main__", "__call_python__"):
    raise SystemExit(main())
