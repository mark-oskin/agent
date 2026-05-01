#!/usr/bin/env -S uv run python3
"""
Terminal UI for multiple embedded agents: pick an agent on the right; each has its own
thinking/tools strip, transcript, and shared prefs-backed wiring.

Requires: ``uv sync --extra tui`` then::

    uv run --extra tui python agent_tui.py

Repeat ``--agent`` for multiple tabs::

    uv run --extra tui python agent_tui.py --agent Planner:llama3.2:latest --agent Coder:qwen2.5-coder:latest

``LABEL:MODEL`` uses that Ollama model; ``LABEL`` alone uses your default model from prefs.

Fork the current agent's history into a new lane (same shared app wiring)::

    /fork Reviewer "Short task one,Short task two"
    /fork Experiment

Quoted part is optional; comma-separated requests run on the new agent in order.

Close an agent by display name (must be unique among lanes)::

    /kill Coder
    /kill "Agent 2"

At least one agent must remain.
"""

from __future__ import annotations

import argparse
import sys
import threading
import traceback
from typing import Callable, Dict, List, Optional, Set, Tuple

from fork_parse import parse_fork_command, parse_kill_command


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
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Footer, Header, Input, OptionList, RichLog, Static
    from textual.widgets.option_list import Option
except ImportError:
    _die_need_tui_extra()


def _is_activity_output_line(text: str) -> bool:
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


def _parse_agent_spec(spec: str) -> Tuple[str, str]:
    """Return (label, ollama_model_or_empty)."""
    s = (spec or "").strip()
    if not s:
        return "Agent", ""
    if ":" in s:
        label, model = s.split(":", 1)
        return label.strip() or "Agent", model.strip()
    return s, ""


class AgentTuiApp(App[None]):
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True),
    ]

    CSS = """
    Screen {
        layout: vertical;
    }
    #main_row {
        height: 1fr;
        layout: horizontal;
        min-height: 12;
    }
    #lanes_container {
        width: 1fr;
        height: 100%;
    }
    .lane-pane {
        layout: vertical;
        height: 100%;
    }
    .lane-pane.hidden {
        display: none;
    }
    .activity_wrap {
        height: 25%;
        min-height: 5;
        border: tall $accent;
        background: $surface;
        layout: vertical;
    }
    .thinking_live {
        height: auto;
        max-height: 55%;
        padding: 0 1;
        background: $surface;
    }
    .activity_log {
        height: 1fr;
        min-height: 3;
        border-top: solid $accent;
    }
    .bottom_lane {
        height: 1fr;
        layout: vertical;
    }
    .stream_box {
        height: auto;
        max-height: 35%;
        border: heavy $boost;
        padding: 0 1;
        background: $panel;
    }
    .chat_log {
        height: 1fr;
        min-height: 8;
        border: tall $surface;
        background: $background;
    }
    #sidebar {
        width: 28;
        height: 100%;
        border-left: tall $accent;
        layout: vertical;
        padding: 0 1;
        background: $panel;
    }
    #sidebar_title {
        height: auto;
        padding: 1 0;
        text-style: bold;
    }
    #agent_list {
        height: 1fr;
    }
    #prompt {
        height: 3;
    }
    """

    def __init__(self, *, verbose: int = 0, agent_specs: Optional[List[str]] = None) -> None:
        super().__init__()
        self._verbose = verbose
        raw = agent_specs if agent_specs else []
        self._specs: List[Tuple[str, str]] = [_parse_agent_spec(x) for x in raw]
        if not self._specs:
            self._specs = [("Agent 1", "")]
        self._n = len(self._specs)
        self._busy_lanes: Set[int] = set()
        self._active_lane = 0
        self._sessions: List = []
        self._embed_app = None

        self._stream_buf: Dict[int, str] = {}
        self._thinking_buf: Dict[int, str] = {}
        self._thinking_follow: Dict[int, bool] = {}
        self._lane_labels: List[str] = [label for label, _ in self._specs]
        # Parallel per-lane widgets (compose mounts synchronously; dynamic fork mounts are deferred).
        self._lane_verticals: List[Vertical] = []
        self._thinking_widgets: List[Static] = []
        self._activity_logs: List[RichLog] = []
        self._stream_widgets: List[Static] = []
        self._chat_logs: List[RichLog] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main_row"):
            with Vertical(id="lanes_container"):
                for i in range(self._n):
                    hidden = "" if i == 0 else " hidden"
                    with Vertical(classes=f"lane-pane{hidden}", id=f"lane-{i}"):
                        with Vertical(classes="activity_wrap"):
                            yield Static("", classes="thinking_live", id=f"thinking-{i}")
                            yield RichLog(
                                classes="activity_log",
                                id=f"activity-{i}",
                                highlight=False,
                                markup=True,
                                wrap=True,
                            )
                        with Vertical(classes="bottom_lane"):
                            yield Static("", classes="stream_box", id=f"stream-{i}")
                            yield RichLog(
                                classes="chat_log",
                                id=f"chat-{i}",
                                highlight=False,
                                markup=True,
                                wrap=True,
                            )
            with Vertical(id="sidebar"):
                yield Static("Agents", id="sidebar_title")
                opts = []
                for i, (label, model_part) in enumerate(self._specs):
                    line = f"{label}" if not model_part else f"{label}\n  {model_part}"
                    opts.append(Option(line, id=f"agent-{i}"))
                yield OptionList(*opts, id="agent_list")
        yield Input(
            id="prompt",
            placeholder="Message or /command… · /fork … · /kill NAME",
        )
        yield Footer()

    def on_mount(self) -> None:
        from agentlib import build_embedded_session
        from agentlib.llm.profile import LlmProfile

        self.title = "Agent TUI"
        ol = self.query_one("#agent_list", OptionList)
        ol.highlighted = 0

        for i, (label, model_part) in enumerate(self._specs):
            prof = LlmProfile(backend="ollama", model=model_part) if model_part.strip() else None
            if i == 0:
                app, sess = build_embedded_session(
                    verbose=int(self._verbose), primary_profile=prof, app=None
                )
                self._embed_app = app
            else:
                _, sess = build_embedded_session(
                    verbose=int(self._verbose), primary_profile=prof, app=self._embed_app
                )
            self._sessions.append(sess)
            self._stream_buf[i] = ""
            self._thinking_buf[i] = ""
            self._thinking_follow[i] = False
            self._lane_verticals.append(self.query_one(f"#lane-{i}", Vertical))
            self._thinking_widgets.append(self.query_one(f"#thinking-{i}", Static))
            self._activity_logs.append(self.query_one(f"#activity-{i}", RichLog))
            self._stream_widgets.append(self.query_one(f"#stream-{i}", Static))
            self._chat_logs.append(self.query_one(f"#chat-{i}", RichLog))
            chat = self._chat_logs[-1]
            hint = (
                f"[bold]{escape(label)}[/bold]"
                + (
                    f" — [dim]{escape(model_part)}[/dim]"
                    if model_part.strip()
                    else " — [dim]default model[/dim]"
                )
                + "\n[dim]Ctrl+Q quit · /help · pick agent on the right[/dim]"
            )
            chat.write(Text.from_markup(hint))

        self._sync_prompt_enabled()

    def action_quit(self) -> None:
        self.exit()

    def _sidebar_line_for_agent(self, label: str, session) -> str:
        from agentlib.llm.profile import effective_ollama_model_from_profile

        pp = session.primary_profile
        if pp is None:
            return label
        if getattr(pp, "backend", "") == "hosted":
            model_part = (getattr(pp, "model", None) or "") or "hosted"
        else:
            assert self._embed_app is not None
            model_part = effective_ollama_model_from_profile(pp, self._embed_app.ollama_model())
        return label if not str(model_part).strip() else f"{label}\n  {model_part}"

    def _mount_lane_widgets(self, idx: int, *, hidden: bool) -> RichLog:
        thinking = Static("", classes="thinking_live", id=f"thinking-{idx}")
        activity = RichLog(
            classes="activity_log",
            id=f"activity-{idx}",
            highlight=False,
            markup=True,
            wrap=True,
        )
        activity_wrap = Vertical(thinking, activity, classes="activity_wrap")
        stream = Static("", classes="stream_box", id=f"stream-{idx}")
        chat = RichLog(
            classes="chat_log",
            id=f"chat-{idx}",
            highlight=False,
            markup=True,
            wrap=True,
        )
        bottom = Vertical(stream, chat, classes="bottom_lane")
        lane_cls = "lane-pane hidden" if hidden else "lane-pane"
        lane = Vertical(activity_wrap, bottom, classes=lane_cls, id=f"lane-{idx}")
        self._lane_verticals.append(lane)
        self._thinking_widgets.append(thinking)
        self._activity_logs.append(activity)
        self._stream_widgets.append(stream)
        self._chat_logs.append(chat)
        container = self.query_one("#lanes_container", Vertical)
        container.mount(lane)
        return chat

    def _handle_fork(self, line: str) -> None:
        from agentlib import fork_embedded_session

        parent_lane = self._active_lane
        parsed = parse_fork_command(line)
        if parsed is None:
            self._activity_logs[parent_lane].write(
                Text.from_markup(
                    "[yellow]/fork NAME[/yellow] or [yellow]/fork NAME \"cmd1,cmd2\"[/yellow]"
                )
            )
            return
        name, cmds = parsed
        parent_sess = self._sessions[parent_lane]
        new_idx = self._n
        new_sess = fork_embedded_session(parent_sess, app=self._embed_app)

        chat = self._mount_lane_widgets(new_idx, hidden=False)
        ol = self.query_one("#agent_list", OptionList)
        ol.add_option(Option(self._sidebar_line_for_agent(name, new_sess), id=f"agent-{new_idx}"))

        self._lane_labels.append(name)
        self._sessions.append(new_sess)
        self._stream_buf[new_idx] = ""
        self._thinking_buf[new_idx] = ""
        self._thinking_follow[new_idx] = False
        self._n += 1

        hint = (
            f"[bold]{escape(name)}[/bold] — [dim]forked from lane {parent_lane + 1}[/dim]\n"
            f"[dim]Ctrl+Q quit · /help · /fork …[/dim]"
        )
        chat.write(Text.from_markup(hint))

        self._chat_logs[parent_lane].write(
            Text.from_markup(f"[dim]Fork → [bold]{escape(name)}[/bold][/dim]\n")
        )

        self._show_lane(new_idx)
        ol.highlighted = new_idx
        self._sync_prompt_enabled()

        filtered = [c.strip() for c in cmds if c.strip()]
        if filtered:
            self._execute_lines_chain(new_idx, filtered)

    def _lanes_matching_name(self, name: str) -> List[int]:
        key = name.casefold().strip()
        return [i for i, lab in enumerate(self._lane_labels) if lab.casefold().strip() == key]

    def _sync_lane_visual_from(self, src: int, dst: int) -> None:
        """Copy transcript/UI state from lane ``src`` onto widgets at lane ``dst``."""
        schat = self._chat_logs[src]
        dchat = self._chat_logs[dst]
        dchat.clear()
        dchat.lines.extend(list(schat.lines))
        dchat._widest_line_width = schat._widest_line_width
        dchat._start_line = schat._start_line
        dchat.virtual_size = schat.virtual_size
        dchat._line_cache.clear()
        dchat.refresh()

        sa = self._activity_logs[src]
        da = self._activity_logs[dst]
        da.clear()
        da.lines.extend(list(sa.lines))
        da._widest_line_width = sa._widest_line_width
        da._start_line = sa._start_line
        da.virtual_size = sa.virtual_size
        da._line_cache.clear()
        da.refresh()

        ss = self._stream_widgets[src]
        ds = self._stream_widgets[dst]
        ds.update(ss.content)

        st = self._thinking_widgets[src]
        dt = self._thinking_widgets[dst]
        dt.update(st.content)

    def _kill_lane_at(self, k: int) -> None:
        """Remove lane index ``k``; compact indices by swapping last lane into ``k`` when needed."""
        last = self._n - 1
        prev_active = self._active_lane
        ol = self.query_one("#agent_list", OptionList)

        if k != last:
            self._sessions[k] = self._sessions[last]
            self._lane_labels[k] = self._lane_labels[last]
            self._sync_lane_visual_from(last, k)

            line = self._sidebar_line_for_agent(self._lane_labels[k], self._sessions[k])
            ol.replace_option_prompt_at_index(k, line)

            self._stream_buf[k] = self._stream_buf[last]
            self._thinking_buf[k] = self._thinking_buf[last]
            self._thinking_follow[k] = self._thinking_follow[last]

        self._sessions.pop()
        self._lane_labels.pop()
        del self._stream_buf[last]
        del self._thinking_buf[last]
        del self._thinking_follow[last]

        self._busy_lanes.discard(last)
        self._busy_lanes.discard(k)

        lane_widget = self._lane_verticals.pop()
        lane_widget.remove()

        self._thinking_widgets.pop()
        self._activity_logs.pop()
        self._stream_widgets.pop()
        self._chat_logs.pop()

        ol.remove_option_at_index(last)
        self._n -= 1

        if prev_active == last:
            self._active_lane = k if k != last else max(0, last - 1)
        else:
            self._active_lane = prev_active
        self._active_lane = max(0, min(self._active_lane, self._n - 1))

        ol.highlighted = self._active_lane
        self._show_lane(self._active_lane)
        self._sync_prompt_enabled()

    def _handle_kill(self, line: str) -> None:
        fb = self._active_lane
        act = self._activity_logs[fb]

        name = parse_kill_command(line)
        if name is None:
            act.write(Text.from_markup("[yellow]/kill NAME[/yellow] or [yellow]/kill \"Long Name\"[/yellow]"))
            return
        matches = self._lanes_matching_name(name)
        if not matches:
            act.write(Text.from_markup(f"[yellow]No agent named[/yellow] [bold]{escape(name)}[/bold]"))
            return
        if len(matches) > 1:
            lanes_s = ", ".join(str(i + 1) for i in matches)
            act.write(
                Text.from_markup(
                    f"[yellow]Ambiguous name[/yellow] [bold]{escape(name)}[/bold] "
                    f"[dim](lanes {lanes_s}); give forks distinct names[/dim]"
                )
            )
            return
        k = matches[0]
        if self._n <= 1:
            act.write(Text.from_markup("[yellow]Cannot remove the last agent.[/yellow]"))
            return

        victim_label = self._lane_labels[k]
        self._kill_lane_at(k)

        ack_lane = self._active_lane
        self._activity_logs[ack_lane].write(
            Text.from_markup(f"[dim]Killed[/dim] [bold]{escape(victim_label)}[/bold]")
        )

    def _execute_lines_chain(self, lane: int, lines: List[str]) -> None:
        if not lines:
            return
        self._set_lane_busy(lane, True)
        session = self._sessions[lane]

        def worker() -> None:
            emit: Callable[[dict], None] = lambda ev, ln=lane: self._emit_for(ln, ev)
            try:
                for i, ln in enumerate(lines):
                    last = i == len(lines) - 1
                    self.call_from_thread(self._prepare_turn_ui, lane, ln)
                    res = session.execute_line(ln, emit=emit)
                    self.call_from_thread(self._apply_turn_result, lane, res, finalize_busy=last)
            except BaseException:
                tb = traceback.format_exc()
                self.call_from_thread(self._turn_error, lane, tb)

        threading.Thread(target=worker, name=f"agent-chain-{lane}", daemon=True).start()

    def _prepare_turn_ui(self, lane: int, line: str) -> None:
        chat = self._chat_logs[lane]
        stream_w = self._stream_widgets[lane]
        thinking_live = self._thinking_widgets[lane]
        chat.write(Text.from_markup(f"[bold green]You[/bold green]\n{escape(line)}\n"))
        self._stream_buf[lane] = ""
        stream_w.update("")
        self._thinking_buf[lane] = ""
        thinking_live.update("")

    def _show_lane(self, index: int) -> None:
        index = max(0, min(index, self._n - 1))
        self._active_lane = index
        for i in range(self._n):
            self._lane_verticals[i].set_classes(f"lane-pane{' hidden' if i != index else ''}")

    @on(OptionList.OptionSelected, "#agent_list")
    def agent_selected(self, event: OptionList.OptionSelected) -> None:
        self._show_lane(event.option_index)
        self._sync_prompt_enabled()

    def _emit_for(self, lane: int, ev: dict) -> None:
        payload = dict(ev)
        self.call_from_thread(self._dispatch_emit, lane, payload)

    def _dispatch_emit(self, lane: int, ev: dict) -> None:
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

        activity = self._activity_logs[lane]
        thinking_live = self._thinking_widgets[lane]
        stream_w = self._stream_widgets[lane]
        chat = self._chat_logs[lane]

        if t == "thinking":
            if "[Thinking]" in text:
                self._thinking_follow[lane] = True
                self._thinking_buf[lane] = ""
            self._thinking_buf[lane] += text
            if "[Done thinking]" in text:
                self._thinking_follow[lane] = False
                activity.write(Text.from_markup(f"[dim]{escape(self._thinking_buf[lane].strip())}[/dim]"))
                self._thinking_buf[lane] = ""
                thinking_live.update("")
                return
            thinking_live.update(Text.from_markup(f"[dim]{escape(self._thinking_buf[lane])}[/dim]"))
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
                self._stream_buf[lane] += text
                buf = self._stream_buf[lane]
                if buf:
                    stream_w.update(
                        Text.from_markup(
                            f"[bold]Assistant[/bold] [dim](stream)[/dim]\n{escape(buf)}"
                        )
                    )
                return
            if self._thinking_follow.get(lane):
                activity.write(Text.from_markup(f"[dim]{escape(text)}[/dim]"))
                return
            if _is_activity_output_line(text):
                activity.write(Text(escape(text)))
            else:
                chat.write(Text.from_markup(f"[dim]{escape(text)}[/dim]"))
            return

        activity.write(Text.from_markup(f"[magenta]{t}[/magenta] {escape(text)}"))

    def _sync_prompt_enabled(self) -> None:
        pr = self.query_one("#prompt", Input)
        blocked = self._active_lane in self._busy_lanes
        pr.disabled = blocked
        if not blocked:
            pr.focus()

    def _set_lane_busy(self, lane: int, busy: bool) -> None:
        if busy:
            self._busy_lanes.add(lane)
        else:
            self._busy_lanes.discard(lane)
        self._sync_prompt_enabled()

    @on(Input.Submitted, "#prompt")
    def submit_prompt(self, event: Input.Submitted) -> None:
        line = (event.value or "").strip()
        event.input.value = ""
        lane = self._active_lane
        if not line or lane >= len(self._sessions):
            return
        if self._active_lane in self._busy_lanes:
            return
        if line.startswith("/kill"):
            self._handle_kill(line)
            return
        if line.startswith("/fork"):
            self._handle_fork(line)
            return
        self._run_line(lane, line)

    def _run_line(self, lane: int, line: str) -> None:
        self._prepare_turn_ui(lane, line)
        self._set_lane_busy(lane, True)

        session = self._sessions[lane]

        def worker() -> None:
            emit: Callable[[dict], None] = lambda ev, ln=lane: self._emit_for(ln, ev)
            try:
                res = session.execute_line(line, emit=emit)
                self.call_from_thread(self._turn_done, lane, res)
            except BaseException:
                tb = traceback.format_exc()
                self.call_from_thread(self._turn_error, lane, tb)

        threading.Thread(target=worker, name=f"agent-turn-{lane}", daemon=True).start()

    def _turn_error(self, lane: int, tb: str) -> None:
        self._set_lane_busy(lane, False)
        activity = self._activity_logs[lane]
        activity.write(Text.from_markup(f"[bold red]Turn error[/bold red]\n{escape(tb)}"))
        self._stream_widgets[lane].update("")
        self._thinking_buf[lane] = ""
        self._thinking_widgets[lane].update("")

    def _turn_done(self, lane: int, res: dict) -> None:
        self._apply_turn_result(lane, res, finalize_busy=True)

    def _apply_turn_result(self, lane: int, res: dict, *, finalize_busy: bool) -> None:
        try:
            if res.get("quit"):
                self.exit()
                return
            chat = self._chat_logs[lane]

            if res.get("type") == "command":
                out = res.get("output") or ""
                if str(out).strip():
                    self._activity_logs[lane].write(
                        Text.from_markup(f"[yellow]{escape(str(out))}[/yellow]")
                    )
                return

            if res.get("type") == "turn":
                ans = res.get("answer")
                if isinstance(ans, str) and ans.strip():
                    chat.write(Text.from_markup(f"[bold cyan]Assistant[/bold cyan]\n{escape(ans)}\n"))
                elif self._stream_buf.get(lane, "").strip():
                    chat.write(
                        Text.from_markup(
                            f"[bold cyan]Assistant[/bold cyan]\n{escape(self._stream_buf[lane])}\n"
                        )
                    )
        finally:
            self._stream_buf[lane] = ""
            self._stream_widgets[lane].update("")
            if self._thinking_buf.get(lane, "").strip():
                self._activity_logs[lane].write(
                    Text.from_markup(f"[dim]{escape(self._thinking_buf[lane].strip())}[/dim]")
                )
            self._thinking_buf[lane] = ""
            self._thinking_widgets[lane].update("")
            if finalize_busy:
                self._set_lane_busy(lane, False)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Multi-agent TUI (embedded sessions + streaming emit).")
    p.add_argument("-v", "--verbose", action="count", default=0)
    p.add_argument(
        "--agent",
        action="append",
        dest="agents",
        metavar="LABEL[:MODEL]",
        help="Agent slot (repeat). Examples: Planner:llama3.2:latest or Research",
    )
    args = p.parse_args(argv)
    AgentTuiApp(verbose=int(args.verbose), agent_specs=args.agents).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
