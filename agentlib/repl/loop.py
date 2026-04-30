"""Main stdin loop for interactive REPL sessions."""

from __future__ import annotations

from typing import Callable

from agentlib.session import AgentSession


def run_interactive_repl_loop(
    session: AgentSession,
    *,
    install_readline: Callable[[], None],
    repl_read_line: Callable[[str], str],
    flush_repl_history: Callable[[], None],
    agent_progress: Callable[[str], None],
    prompt: str = "> ",
) -> None:
    install_readline()
    print("Interactive mode. Type /help for commands.")
    while True:
        try:
            line = repl_read_line(prompt)
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\n[Cancelled]\n")
            continue
        s = line.strip()
        if not s:
            continue
        try:
            res = session.execute_line(s)
        except KeyboardInterrupt:
            agent_progress("Cancelled current request (Ctrl-C).")
            print("\n[Cancelled]\n")
            continue
        if res.quit:
            break
        if res.output:
            print(res.output)

    flush_repl_history()
