"""Main stdin loop for interactive REPL sessions."""

from __future__ import annotations

from typing import Callable, Optional

from agentlib.repl.io import read_repl_lines_until_balanced_triple_double_quotes
from agentlib.session import AgentSession


def run_interactive_repl_loop(
    session: AgentSession,
    *,
    install_readline: Callable[[], None],
    repl_read_line: Callable[[str], str],
    flush_repl_history: Callable[[], None],
    agent_progress: Callable[[str], None],
    prompt: str = "> ",
    max_input_bytes: int,
    repl_commit_input_history: Optional[Callable[[str], None]] = None,
) -> None:
    install_readline()
    print("Interactive mode. Type /help for commands.")
    while True:
        try:
            block = read_repl_lines_until_balanced_triple_double_quotes(
                repl_read_line,
                first_prompt=prompt,
                continuation_prompt="... ",
                max_bytes=max_input_bytes,
                repl_commit_history=repl_commit_input_history,
            )
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\n[Cancelled]\n")
            continue
        except ValueError as e:
            print(f"\n{e}\n", flush=True)
            continue
        s = block.strip()
        if not s:
            continue
        try:
            res = session.execute_line(s)
        except KeyboardInterrupt:
            agent_progress("Cancelled current request (Ctrl-C).")
            print("\n[Cancelled]\n")
            continue
        if res.get("quit"):
            break
        out = res.get("output") or ""
        if out:
            print(out)
        pref = res.get("prefill_prompt")
        if isinstance(pref, str) and pref.strip():
            print(pref)
        # For normal turns, `execute_line` returns the answer in the payload.
        ans = res.get("answer")
        if isinstance(ans, str) and ans.strip():
            print(ans)

    flush_repl_history()
