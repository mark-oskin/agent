#!/usr/bin/env -S uv run python3
"""
Terminal UI for the embedded agent: ~25% activity (thinking, tools, warnings) and ~75% chat + input.

Requires optional deps: ``uv sync --extra tui`` then::

    uv run --extra tui python agent_tui.py

Or: ``pip install 'agent[tui]'`` from this project root.
"""

from __future__ import annotations

import argparse
import sys
import threading
import traceback
from typing import Optional

def _die_need_tui_extra() -> None:
    sys.stderr.write(
        "Missing Textual (TUI). Install with:\n"
        "  uv sync --extra tui\n"
        "then run:\n"
        "  uv run --extra tui python agent_tui.py\n"
    )
    raise SystemExit(1)


try:
    from rich.markup import escape
    from rich.text import Text
    from textual import on
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Vertical
    from textual.widgets import Footer, Header, Input, RichLog, Static
except ImportError:
    _die_need_tui_extra()


def _is_activity_output_line(text: str) -> bool:
    """Heuristic: lines that belong in the activity pane vs assistant chat."""
    s = (text or "").strip()
    if not s:
        return False
    if s.startswith("[*]"):
        return True
    if s.startswith("→ ") or s.startswith("→\t"):
        return True
    if s.startswith("Runner:"):
        return True
    if "[skills:" in s:
        return True
    if s.startswith("Tool `") and "finished." in s:
        return True
    return False


class AgentTuiApp(App[None]):
    """Split-pane agent front-end driven by ``execute_line(..., emit=...)``."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #activity_wrap {
        height: 25%;
        min-height: 5;
        border: tall $accent;
        background: $surface;
        layout: vertical;
    }
    #thinking_live {
        height: auto;
        max-height: 55%;
        padding: 0 1;
        background: $surface;
    }
    #activity {
        height: 1fr;
        min-height: 3;
        border-top: solid $accent;
    }
    #bottom {
        height: 1fr;
        layout: vertical;
    }
    #stream {
        height: auto;
        max-height: 35%;
        border: heavy $boost;
        padding: 0 1;
        background: $panel;
    }
    #chat {
        height: 1fr;
        min-height: 8;
        border: tall $surface;
        background: $background;
    }
    #prompt {
        height: 3;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True),
    ]

    def __init__(self, *, verbose: int = 0) -> None:
        super().__init__()
        self._verbose = verbose
        self._busy = False
        self._stream_buf = ""
        self._session = None
        self._thinking_follow = False
        self._thinking_buf = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="activity_wrap"):
            yield Static("", id="thinking_live")
            yield RichLog(id="activity", highlight=False, markup=True, wrap=True)
        with Vertical(id="bottom"):
            yield Static("", id="stream")
            yield RichLog(id="chat", highlight=False, markup=True, wrap=True)
            yield Input(id="prompt", placeholder="Message or /command… (Enter to send)")
        yield Footer()

    def on_mount(self) -> None:
        from agentlib import build_embedded_session

        self.title = "Agent TUI"
        _, self._session = build_embedded_session(verbose=int(self._verbose))
        chat = self.query_one("#chat", RichLog)
        chat.write(
            Text.from_markup(
                "[bold]Agent TUI[/bold] — top: thinking / tools / warnings; bottom: replies. "
                "[dim]Ctrl+Q quit · /help for commands[/dim]"
            )
        )
        self.query_one("#prompt", Input).focus()

    def action_quit(self) -> None:
        self.exit()

    @on(Input.Submitted, "#prompt")
    def submit_prompt(self, event: Input.Submitted) -> None:
        line = (event.value or "").strip()
        event.input.value = ""
        if not line or self._busy or self._session is None:
            return
        self._run_line(line)

    def _emit(self, ev: dict) -> None:
        payload = dict(ev)
        self.call_from_thread(self._dispatch_emit, payload)

    def _dispatch_emit(self, ev: dict) -> None:
        t = ev.get("type") or "output"
        text = ev.get("text")
        if text is None:
            text = ""
        elif not isinstance(text, str):
            text = str(text)
        end = ev.get("end", "\n")
        partial = bool(ev.get("partial"))

        if not text and end == "\n" and not partial:
            return

        activity = self.query_one("#activity", RichLog)
        thinking_live = self.query_one("#thinking_live", Static)
        stream_w = self.query_one("#stream", Static)
        chat = self.query_one("#chat", RichLog)

        if t == "thinking":
            if "[Thinking]" in text:
                self._thinking_follow = True
                self._thinking_buf = ""
            self._thinking_buf += text
            if "[Done thinking]" in text:
                self._thinking_follow = False
                activity.write(Text.from_markup(f"[dim]{escape(self._thinking_buf.strip())}[/dim]"))
                self._thinking_buf = ""
                thinking_live.update("")
                return
            thinking_live.update(Text.from_markup(f"[dim]{escape(self._thinking_buf)}[/dim]"))
            return

        if t in ("progress", "warning", "stderr", "debug", "error"):
            style = "yellow" if t == "warning" else "red" if t == "error" else "dim"
            activity.write(Text.from_markup(f"[{style}]{escape(text)}[/{style}]"))
            return

        if t == "answer":
            chat.write(Text.from_markup(f"[bold cyan]Assistant[/bold cyan]\n{escape(text)}\n"))
            return

        if t == "output":
            if partial:
                self._stream_buf += text
                if self._stream_buf:
                    stream_w.update(
                        Text.from_markup(
                            f"[bold]Assistant[/bold] [dim](stream)[/dim]\n{escape(self._stream_buf)}"
                        )
                    )
                return
            if self._thinking_follow:
                activity.write(Text.from_markup(f"[dim]{escape(text)}[/dim]"))
                return
            if _is_activity_output_line(text):
                activity.write(Text(escape(text)))
            else:
                chat.write(Text.from_markup(f"[dim]{escape(text)}[/dim]"))
            return

        activity.write(Text.from_markup(f"[magenta]{t}[/magenta] {escape(text)}"))

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        pr = self.query_one("#prompt", Input)
        pr.disabled = busy
        if not busy:
            pr.focus()

    def _run_line(self, line: str) -> None:
        chat = self.query_one("#chat", RichLog)
        stream_w = self.query_one("#stream", Static)
        chat.write(Text.from_markup(f"[bold green]You[/bold green]\n{escape(line)}\n"))
        self._stream_buf = ""
        stream_w.update("")
        self._thinking_buf = ""
        self.query_one("#thinking_live", Static).update("")
        self._set_busy(True)

        session = self._session
        emit = self._emit

        def worker() -> None:
            try:
                res = session.execute_line(line, emit=emit)
                self.call_from_thread(self._turn_done, res)
            except BaseException:
                tb = traceback.format_exc()
                self.call_from_thread(self._turn_error, tb)

        threading.Thread(target=worker, name="agent-turn", daemon=True).start()

    def _turn_error(self, tb: str) -> None:
        self._set_busy(False)
        activity = self.query_one("#activity", RichLog)
        activity.write(Text.from_markup(f"[bold red]Turn error[/bold red]\n{escape(tb)}"))
        self.query_one("#stream", Static).update("")
        self._thinking_buf = ""
        self.query_one("#thinking_live", Static).update("")

    def _turn_done(self, res: dict) -> None:
        try:
            if res.get("quit"):
                self.exit()
                return
            chat = self.query_one("#chat", RichLog)

            if res.get("type") == "command":
                out = res.get("output") or ""
                if str(out).strip():
                    self.query_one("#activity", RichLog).write(
                        Text.from_markup(f"[yellow]{escape(str(out))}[/yellow]")
                    )
                return

            if res.get("type") == "turn":
                ans = res.get("answer")
                if isinstance(ans, str) and ans.strip():
                    chat.write(Text.from_markup(f"[bold cyan]Assistant[/bold cyan]\n{escape(ans)}\n"))
                elif self._stream_buf.strip():
                    chat.write(
                        Text.from_markup(
                            f"[bold cyan]Assistant[/bold cyan]\n{escape(self._stream_buf)}\n"
                        )
                    )
        finally:
            self._stream_buf = ""
            self.query_one("#stream", Static).update("")
            if self._thinking_buf.strip():
                self.query_one("#activity", RichLog).write(
                    Text.from_markup(f"[dim]{escape(self._thinking_buf.strip())}[/dim]")
                )
            self._thinking_buf = ""
            self.query_one("#thinking_live", Static).update("")
            self._set_busy(False)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Agent TUI (embedded session + streaming emit).")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Verbose agent / tool heartbeats")
    args = p.parse_args(argv)
    AgentTuiApp(verbose=int(args.verbose)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
