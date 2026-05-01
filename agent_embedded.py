#!/usr/bin/env -S uv run python

from __future__ import annotations

import argparse
import sys
from typing import Callable, Optional


def make_demo_emitter() -> Callable[[dict], None]:
    """
    Example emit callback that splits events by type.

    - thinking/progress/warning go to stderr
    - output goes to stdout
    """

    in_thinking = False

    def emit(ev: dict) -> None:
        t = ev.get("type")
        text = ev.get("text") or ""
        if not isinstance(text, str):
            text = str(text)
        end = ev.get("end", "\n")
        flush = bool(ev.get("flush", True))
        if not text and end == "\n":
            return
        nonlocal in_thinking
        if t == "thinking":
            # Markers come through as separate lines; treat following lines as thinking
            # until we see [Done thinking].
            if "[Thinking]" in text:
                in_thinking = True
            if "[Done thinking]" in text:
                in_thinking = False
            print(text, file=sys.stderr, end=end, flush=flush)
            return

        if t in ("progress", "warning"):
            print(text, file=sys.stderr, end=end, flush=flush)
        elif t == "stderr":
            print(text, file=sys.stderr, end=end, flush=flush)
        elif in_thinking:
            print(text, file=sys.stderr, end=end, flush=flush)
        else:
            print(text, end=end, flush=flush)

    return emit


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Embedded AgentSession demo (emit + structured results).")
    p.add_argument("-v", "--verbose", action="count", default=0)
    p.add_argument("--once", nargs=argparse.REMAINDER, help="Run one line and exit (e.g. --once 2+2)")
    args = p.parse_args(argv)

    from agentlib import build_embedded_session

    _app, session = build_embedded_session(verbose=int(args.verbose))
    emit = make_demo_emitter()

    if args.once is not None:
        line = " ".join(args.once).strip()
        res = session.execute_line(line, emit=emit)
        # Embedders can inspect the structured result directly:
        if res.get("type") == "turn":
            print(f"[result] answered={res.get('answered')} answer={res.get('answer')!r}")
        else:
            print(f"[result] {res}")
        return 0

    print("Embedded mode. Type /help for commands; Ctrl-D to exit.")
    while True:
        try:
            line = input("> ")
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\n[Cancelled]\n", file=sys.stderr)
            continue
        res = session.execute_line(line, emit=emit)
        if res.get("quit"):
            break
        # In an embedding scenario, you likely render `res["answer"]` in your UI:
        ans = res.get("answer")
        if isinstance(ans, str) and ans.strip():
            print(ans)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

