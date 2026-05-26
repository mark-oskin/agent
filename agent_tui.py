#!/usr/bin/env -S uv run python3
"""
Terminal UI for multiple embedded agents: pick an agent on the right; each lane has a
main transcript plus an optional large thinking panel (shown only while reasoning).
Tool and status lines go to the transcript.

Requires: ``uv sync --extra tui`` then::

    uv run --extra tui python agent_tui.py

Repeat ``--agent`` for multiple tabs::

    uv run --extra tui python agent_tui.py --agent Planner:llama3.2:latest --agent Coder:qwen2.5-coder:latest

``LABEL:MODEL`` uses that Ollama model; ``LABEL`` alone uses your default model from prefs.

Fork the current agent's history into a new lane (same shared app wiring)::

    /fork Reviewer "Short task one,Short task two"
    /fork Experiment
    /fork_background Worker "queued task"

``/fork_background`` creates the lane but keeps the current agent focused; comma-separated
requests still run on the new lane when quoted as with ``/fork``.

Close an agent by display name (must be unique among lanes)::

    /kill Coder
    /kill "Agent 2"

At least one agent must remain.

Run Python in-process (same helpers as CLI): ``/call_python help`` — ``ai()`` targets this lane;
``ai(cmd, name)`` and ``fork_agent()`` target other lanes when using multi-agent hooks.

Inspect or jump lanes without running the model::

    /list
    /switch Planner
    /send Coder /help                    # async on Coder (queues if that lane is busy)
    /turn self What is 2+2?              # blocking on this lane; returns the answer
    /send self follow-up question        # async after the current turn on this lane finishes
    /send Worker "/help,/show model"     # comma-separated list in "…"; use '…' or \, inside for literal commas (/fork rules)

    /last answer|question [NAME]   (aliases: /last_answer, /last_question)

``list_agents()``, ``switch_agent(...)``, ``send(...)``, ``last_answer(...)``, ``last_question(...)`` inside ``/call_python``
mirror those behaviors for Telegram bridges and scripts.

Run a shell command locally like the agent ``run_command`` tool: ``/run_command help`` or ``! ls``.

Clipboard: ``/clipboard copy|copy all|paste`` (`paste` loads the clipboard into your prompt so you can edit before Enter). Session JSON: ``/context load|save|start_log FILE`` (``/save_context`` is a one-shot snapshot; ``start_log`` enables auto-save after each turn). Extensions: ``/load FILE.py`` (``register_repl``), ``/unload``, ``/extensions``.

**Mouse:** two-finger scroll in the chat or thinking log to move through output; drag to select text (within that pane only); releasing the button copies to the clipboard (macOS Terminal usually does not pass **Cmd+C** through). **Ctrl+C** or **Ctrl+Shift+C** also copies when idle; **Cmd+C** works in terminals that forward it (iTerm, Ghostty, WezTerm). The prompt uses normal editor copy/paste (**Ctrl+V** / **Cmd+V** to paste).

Prompt history is **per lane**: **↑** / **↓** when the cursor is on the **first / last** line recall prior messages (like the CLI);
otherwise they move inside the editor. **Ctrl+↑** / **Ctrl+↓** always recall (even mid‑multiline). **Enter** sends the message (same idea as the single-line input).
For an extra line inside the box use **Shift+Enter** or **Ctrl+J**, or paste multiline text; content scrolls vertically
when it does not fit.

The **first startup agent** (the initial ``Agent 1`` label unless you passed ``--agent``) loads and appends to the same
``~/.agent_repl_history`` file as the CLI; other agents/forks only keep in-memory recall for that lane.

While a turn is running, **Ctrl+C** opens a short prompt (**not** the old toast): press **Ctrl+C** again to send an
interrupt to the worker (best-effort cancel), or press **any other key** to close the prompt and keep waiting.
**Ctrl+Q** still quits the app when no modal is open (see Textual's quit hint when idle).
"""

from __future__ import annotations

import warnings

# Emitted from urllib3/__init__.py on macOS LibreSSL; filter before anything imports urllib3.
warnings.filterwarnings(
    "ignore",
    message=r"urllib3 v2 only supports OpenSSL.*",
    category=Warning,
    module=r"urllib3(\..*)?",
)

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from agentlib.llm.gen_rate import GenRateTracker, estimate_tokens_from_text
from agentlib.tui_widgets import (
    AgentScreen,
    NoSelectStatic,
    SelectableRichLog,
    is_copy_selection_key,
)
from agentlib.tui_parse import (
    parse_fork_background_command,
    parse_fork_command,
    parse_kill_command,
    parse_send_command,
    parse_turn_command,
)


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
    from textual import events, on
    from textual.actions import SkipAction
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.screen import Screen
    from textual.widgets import Footer, Header, OptionList, RichLog, Static, TextArea
    from textual.widgets._header import HeaderIcon
    from textual.widgets.option_list import Option
except ImportError:
    _die_need_tui_extra()


# Readline hooks the controlling tty when used in-process and breaks Textual key decoding (^[[A leaks as text).


def _decode_readline_history_line(raw: str) -> str:
    """Decode one `_HiStOrY_V2_` line (GNU readline backslash + octal escapes)."""

    out: List[str] = []
    i = 0
    while i < len(raw):
        if raw[i] == "\\" and i + 1 < len(raw):
            c = raw[i + 1]
            if c == "n":
                out.append("\n")
                i += 2
                continue
            if c == "r":
                out.append("\r")
                i += 2
                continue
            if c == "t":
                out.append("\t")
                i += 2
                continue
            if c == "\\":
                out.append("\\")
                i += 2
                continue
            if c in "01234567":
                j = i + 1
                oct_digits = ""
                while j < len(raw) and len(oct_digits) < 3 and raw[j] in "01234567":
                    oct_digits += raw[j]
                    j += 1
                if oct_digits:
                    out.append(chr(int(oct_digits, 8)))
                    i = j
                    continue
        out.append(raw[i])
        i += 1
    return "".join(out)


_REPL_HIST_READ_PY = r"""import readline, sys
p = sys.argv[1]
try:
    readline.read_history_file(p)
except FileNotFoundError:
    pass
for i in range(1, readline.get_history_length() + 1):
    print(readline.get_history_item(i))
"""
_REPL_HIST_APPEND_PY = r"""import json, readline, sys
path, line = json.loads(sys.stdin.buffer.read().decode("utf-8"))
if not line:
    raise SystemExit(0)
try:
    readline.read_history_file(path)
except FileNotFoundError:
    pass
readline.add_history(line)
readline.write_history_file(path)
"""
_REPL_HIST_FLUSH_PY = r"""import readline, sys
p = sys.argv[1]
try:
    readline.read_history_file(p)
except FileNotFoundError:
    pass
readline.write_history_file(p)
"""


def _repl_history_read_lines_subprocess(path: str) -> List[str]:
    """Fallback: load via subprocess readline (often fails on macOS libedit + `_HiStOrY_V2_` files)."""
    path = os.path.abspath(os.path.expanduser(path))
    try:
        out = subprocess.check_output(
            [sys.executable, "-c", _REPL_HIST_READ_PY, path],
            text=True,
            timeout=120,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    return [ln for ln in out.splitlines()]


def _repl_history_read_lines(path: str) -> List[str]:
    """Load `~/.agent_repl_history` without importing readline in the TUI process.

    GNU **\_HiStOrY\_V2\_** files are used by the CLI; macOS libedit often cannot parse them
    (`read_history_file` leaves length -1), so we decode the text format directly.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        return []
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = text.splitlines()
    if not lines:
        return []
    if lines[0].startswith("_HiStOrY_V2"):
        decoded = [_decode_readline_history_line(raw) for raw in lines[1:] if raw.strip()]
        if decoded:
            return decoded
    # Older/plain files: one entry per line; skip timestamp-only meta lines.
    ts_line = re.compile(r"^#\d+\s*$")
    plain: List[str] = []
    for raw in lines:
        s = raw.strip("\r\n")
        if not s.strip():
            continue
        if ts_line.match(s.strip()):
            continue
        plain.append(s)
    if plain:
        return plain
    return _repl_history_read_lines_subprocess(str(p))


def _repl_history_append_line(path: str, line: str) -> None:
    try:
        subprocess.run(
            [sys.executable, "-c", _REPL_HIST_APPEND_PY],
            input=json.dumps([path, line]).encode("utf-8"),
            timeout=120,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass


def _repl_history_flush_file(path: str) -> None:
    try:
        subprocess.run(
            [sys.executable, "-c", _REPL_HIST_FLUSH_PY, path],
            timeout=120,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass


def _set_prompt_text(pr: TextArea, s: str) -> None:
    """Replace prompt text and move the cursor to the end (or home when empty)."""
    pr.text = s
    if not s:
        pr.move_cursor((0, 0))
        return
    li = pr.document.line_count - 1
    pr.move_cursor((li, len(pr.document[li])))


class PromptTextArea(TextArea):
    """Multiline prompt with **Enter to send** (like ``Input``).

    Textual's ``Input`` widget draws only one terminal row—extra CSS height stays blank—so
    a taller box has to be ``TextArea``. We handle **Enter** here instead of inserting ``\\n``.
    """

    _NEWLINE_KEYS = frozenset({"shift+enter", "ctrl+j"})

    async def _on_key(self, event: events.Key) -> None:
        key = event.key
        if is_copy_selection_key(key):
            event.stop()
            event.prevent_default()
            try:
                self.app.action_copy_mouse_selection()
            except SkipAction:
                pass
            return
        if key == "enter":
            event.stop()
            event.prevent_default()
            try:
                self.app.action_submit_prompt_message()
            except SkipAction:
                pass
            return
        if key in self._NEWLINE_KEYS:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        # Readline-style history when cursor is on the first/last document line.
        if key == "up" and self.cursor_at_first_line:
            try:
                self.app.action_prompt_hist_prev()
            except SkipAction:
                await super()._on_key(event)
            else:
                event.stop()
                event.prevent_default()
            return
        if key == "down" and self.cursor_at_last_line:
            try:
                self.app.action_prompt_hist_next()
            except SkipAction:
                await super()._on_key(event)
            else:
                event.stop()
                event.prevent_default()
            return
        await super()._on_key(event)


def _inject_keyboard_interrupt(thread: threading.Thread) -> bool:
    """Raise KeyboardInterrupt inside ``thread`` (best-effort cooperative cancel)."""
    tid = thread.ident
    if tid is None or not thread.is_alive():
        return False
    n = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), ctypes.py_object(KeyboardInterrupt))
    if n == 0:
        return False
    if n != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)
        return False
    return True


class _BusyInterruptScreen(Screen[Optional[bool]]):
    """First Ctrl+C opens this; Ctrl+C again confirms cancel; any other key dismisses."""

    CSS = """
    _BusyInterruptScreen {
        align: center middle;
    }
    #busy_interrupt_box {
        width: 72;
        max-width: 95%;
        height: auto;
        border: heavy $accent;
        padding: 1 2;
        background: $surface;
    }
    """

    def __init__(self, lane_index: int) -> None:
        super().__init__()
        self._lane_index = lane_index

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(
                "[bold]Request still running on this lane[/bold]\n\n"
                "[cyan]Ctrl+C[/cyan] again — interrupt / cancel this turn\n"
                "Any other key — close this prompt and keep waiting\n\n"
                "[dim]Ctrl+Q[/dim] — quit the app when not in this prompt",
                id="busy_interrupt_msg",
            ),
            id="busy_interrupt_box",
        )

    def on_mount(self) -> None:
        self.focus()

    def on_key(self, event: events.Key) -> None:
        key = event.key or ""
        if key == "ctrl+c":
            # Cancel is handled by MainApp.action_interrupt_prompt (same binding, higher dispatch).
            return
        self.dismiss(False)
        event.stop()
        event.prevent_default()


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
        Binding("ctrl+c", "interrupt_prompt", "", show=False, priority=True),
        Binding(
            "ctrl+shift+c,super+c,meta+c",
            "copy_mouse_selection",
            "Copy",
            show=False,
            priority=True,
        ),
        # Up/Down are used by the multiline prompt; use Ctrl+arrows for per-lane history.
        Binding("ctrl+up", "prompt_hist_prev", "", show=False, priority=True),
        Binding("ctrl+down", "prompt_hist_next", "", show=False, priority=True),
    ]

    CSS = """
    /* Textual HeaderIcon defaults to width 8 + horizontal padding; tok/s was clipped ("/" visible, "s" gone). */
    #app_header HeaderIcon {
        width: auto;
        min-width: 11;
        padding: 0 1;
    }
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
    .thinking_panel {
        display: none;
        height: 0;
        min-height: 0;
        layout: vertical;
        border: tall $accent;
        background: $surface;
    }
    .lane-pane.thinking_open .thinking_panel {
        display: block;
        height: 67%;
        min-height: 10;
    }
    .thinking_title {
        height: 1;
        padding: 0 1;
        text-style: bold dim;
    }
    .thinking_log {
        height: 1fr;
        min-height: 6;
    }
    .chat_lane {
        height: 1fr;
        layout: vertical;
        min-height: 8;
    }
    .lane-pane.thinking_open .chat_lane {
        height: 33%;
        min-height: 6;
    }
    .chat_log {
        height: 1fr;
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
        overflow-x: hidden;
    }
    #agent_list > .option-list--option {
        text-overflow: ellipsis;
    }
    /* TextArea draws a tall border (~2 rows); outer height must include it or only ~1 text row fits. */
    #prompt {
        height: 5;
        min-height: 5;
        max-height: 5;
    }
    """

    def __init__(
        self,
        *,
        verbose: int = 0,
        agent_specs: Optional[List[str]] = None,
        debug_llm_log_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        from agentlib.debug_llm_log import set_debug_llm_log_path

        set_debug_llm_log_path(debug_llm_log_path)
        self._verbose = verbose
        raw = agent_specs if agent_specs else []
        self._specs: List[Tuple[str, str]] = [_parse_agent_spec(x) for x in raw]
        if not self._specs:
            self._specs = [("Agent 1", "")]
        self._n = len(self._specs)
        self._busy_lanes: Set[int] = set()
        self._lane_turn_queues: Dict[int, List[str]] = {}
        self._active_lane = 0
        self._sessions: List = []
        self._embed_app = None

        self._chat_live_buf: Dict[int, str] = {}
        self._chat_stream_open: Dict[int, bool] = {}
        # RichLog.lines index where the in-progress assistant block starts (one write() may add many lines).
        self._chat_stream_line_start: Dict[int, int] = {}
        # Monotonic per-lane turn id; stale workers must not finalize or reset a newer turn's stream.
        self._lane_turn_seq: Dict[int, int] = {}
        self._lane_gen_tok_s: Dict[int, float] = {}
        self._lane_rate_tracker: Dict[int, GenRateTracker] = {}
        self._lane_gen_rate_paint_at: Dict[int, float] = {}
        self._header_gen_rate_interval_s = 1.0
        self._thinking_buf: Dict[int, str] = {}
        self._thinking_follow: Dict[int, bool] = {}
        self._lane_labels: List[str] = [label for label, _ in self._specs]
        # Parallel per-lane widgets (compose mounts synchronously; dynamic fork mounts are deferred).
        self._lane_verticals: List[Vertical] = []
        self._thinking_logs: List[SelectableRichLog] = []
        self._chat_logs: List[SelectableRichLog] = []
        self._prompt_hist_lines: Dict[int, List[str]] = {}
        self._prompt_hist_idx: Dict[int, Optional[int]] = {}
        self._lane_worker_threads: Dict[int, threading.Thread] = {}
        # Case-folded label of the first `--agent` slot (default "Agent 1"): shares ~/.agent_repl_history with the CLI.
        self._startup_agent_label_key: str = self._lane_labels[0].casefold().strip()

    def get_default_screen(self) -> Screen:
        return AgentScreen(id="_default")

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True, icon="", id="app_header")
        with Horizontal(id="main_row"):
            with Vertical(id="lanes_container"):
                for i in range(self._n):
                    hidden = "" if i == 0 else " hidden"
                    with Vertical(classes=f"lane-pane{hidden}", id=f"lane-{i}"):
                        with Vertical(classes="thinking_panel", id=f"thinking-panel-{i}"):
                            yield NoSelectStatic("Thinking", classes="thinking_title")
                            yield SelectableRichLog(
                                classes="thinking_log",
                                id=f"thinking-{i}",
                                highlight=False,
                                markup=True,
                                wrap=True,
                                auto_scroll=True,
                            )
                        with Vertical(classes="chat_lane", id=f"chat-lane-{i}"):
                            yield SelectableRichLog(
                                classes="chat_log",
                                id=f"chat-{i}",
                                highlight=False,
                                markup=True,
                                wrap=True,
                                auto_scroll=True,
                            )
            with Vertical(id="sidebar"):
                yield NoSelectStatic("Agents", id="sidebar_title")
                opts = [
                    Option(self._sidebar_line_for_lane(i), id=f"agent-{i}")
                    for i in range(self._n)
                ]
                yield OptionList(*opts, id="agent_list", compact=True)
        yield PromptTextArea(
            "",
            id="prompt",
            placeholder=(
                "Message · Enter send · Shift+Enter or Ctrl+J newline · ↑/↓ history (first/last line) · Ctrl+↑/↓ · "
                "/list · /switch · /send · /fork · /kill · /call_python · ! …"
            ),
            show_line_numbers=False,
            soft_wrap=True,
            tab_behavior="focus",
        )
        yield Footer()

    def on_mount(self) -> None:
        from agentlib import build_embedded_session
        from agentlib.llm.profile import LlmProfile
        from tools import lanes as lanes_tools

        self.title = "Agent TUI"
        self.sub_title = ""
        header_icon = self.query_one("#app_header", Header).query_one(HeaderIcon)
        header_icon.disabled = True
        self.query_one("#app_header", Header).ALLOW_SELECT = False
        self.query_one(Footer).ALLOW_SELECT = False
        for node in self.query(OptionList):
            node.ALLOW_SELECT = False
        ol = self.query_one("#agent_list", OptionList)
        ol.highlighted = 0

        # Wire the lanes plugin tool (agent_send) to the TUI host bridges.
        lanes_tools.set_lanes_host(
            enqueue_line=self._python_enqueue_bridge,
            delegate_line=self._python_delegate_bridge,
        )
        for i, (label, model_part) in enumerate(self._specs):
            prof = LlmProfile(backend="ollama", model=model_part) if model_part.strip() else None
            py_kw = dict(
                python_fork_agent=(lambda li=i: lambda name, cmds=None: self._fork_lane_hook(li, name, cmds, background=False))(),
                python_fork_background_agent=(lambda li=i: lambda name, cmds=None: self._fork_lane_hook(li, name, cmds, background=True))(),
                python_delegate_line=self._python_delegate_bridge,
                python_enqueue_line=self._python_enqueue_bridge,
                python_host_command=self._python_host_bridge,
            )
            if i == 0:
                app, sess = build_embedded_session(
                    verbose=int(self._verbose),
                    primary_profile=prof,
                    app=None,
                    **py_kw,
                )
                self._embed_app = app
            else:
                _, sess = build_embedded_session(
                    verbose=int(self._verbose),
                    primary_profile=prof,
                    app=self._embed_app,
                    **py_kw,
                )
            self._sessions.append(sess)
            # Enable cross-lane tool in TUI sessions (tool policy + prompt docs).
            sess.enabled_toolsets.add("lanes")
            sess.enabled_tools.add("agent_send")
            self._chat_live_buf[i] = ""
            self._chat_stream_open[i] = False
            self._lane_turn_seq[i] = 0
            self._thinking_buf[i] = ""
            self._thinking_follow[i] = False
            self._lane_verticals.append(self.query_one(f"#lane-{i}", Vertical))
            self._thinking_logs.append(self.query_one(f"#thinking-{i}", SelectableRichLog))
            self._chat_logs.append(self.query_one(f"#chat-{i}", SelectableRichLog))
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

        self._hydrate_startup_prompt_history_from_disk()
        self._sync_prompt_enabled()
        for i in range(self._n):
            self._refresh_sidebar_lane(i)

    def _lane_matches_startup_disk_sync(self, lane: int) -> bool:
        if lane < 0 or lane >= len(self._lane_labels):
            return False
        return self._lane_labels[lane].casefold().strip() == self._startup_agent_label_key

    def _hydrate_startup_prompt_history_from_disk(self) -> None:
        """Load CLI ``~/.agent_repl_history`` into recall lists for lanes whose label matches startup."""
        app = self._embed_app
        if app is None:
            return
        lines = _repl_history_read_lines(app.repl_history_path())
        if not lines:
            return
        for lane in range(self._n):
            if self._lane_matches_startup_disk_sync(lane):
                self._prompt_hist_lines[lane] = list(lines)

    @staticmethod
    def _repl_history_one_line(text: str) -> str:
        """Normalize for GNU readline history (single line)."""
        return " ".join((text or "").splitlines()).strip()

    def _append_startup_lane_submission_to_disk(self, lane: int, text: str) -> None:
        if not self._lane_matches_startup_disk_sync(lane):
            return
        app = self._embed_app
        if app is None:
            return
        line = self._repl_history_one_line(text)
        if not line:
            return
        _repl_history_append_line(app.repl_history_path(), line)

    def _flush_disk_repl_history(self) -> None:
        app = self._embed_app
        if app is None:
            return
        _repl_history_flush_file(app.repl_history_path())

    def exit(self, result=None, return_code: int = 0, message=None) -> None:
        self._flush_disk_repl_history()
        super().exit(result, return_code=return_code, message=message)

    def action_interrupt_prompt(self) -> None:
        """Ctrl+C: cancel busy turn, or copy selection when idle."""
        # App-level ctrl+c binding runs before the modal's on_key; do not no-op here or the
        # second Ctrl+C never reaches dismiss(True).
        if isinstance(self.screen, _BusyInterruptScreen):
            self.screen.dismiss(True)
            return
        lane = self._active_lane
        if lane in self._busy_lanes:
            self.push_screen(
                _BusyInterruptScreen(lane),
                callback=lambda r, ln=lane: self._on_busy_interrupt_result(ln, r),
            )
            return
        self.action_copy_mouse_selection()

    async def on_event(self, event: events.Event) -> None:
        if isinstance(event, events.Key) and not event.is_forwarded:
            if is_copy_selection_key(event.key):
                for log in self.query(SelectableRichLog):
                    if log.has_text_selection() and log.copy_selection_to_clipboard():
                        return
        await super().on_event(event)

    def action_copy_mouse_selection(self) -> None:
        """Copy log selection or prompt selection to the OS clipboard."""
        from agentlib.clipboard_io import ClipboardError, clipboard_write_text

        for log in self.query(SelectableRichLog):
            if log.copy_selection_to_clipboard():
                return
        pr = self.query_one("#prompt", TextArea)
        if self.screen.focused is pr:
            text = pr.selected_text
            if text:
                try:
                    clipboard_write_text(text)
                except ClipboardError as e:
                    self.notify(str(e), title="Clipboard", severity="error", timeout=4)
                return
        raise SkipAction()

    def _on_busy_interrupt_result(self, lane: int, result: Optional[bool]) -> None:
        if result is not True:
            return
        th = self._lane_worker_threads.get(lane)
        if th is None or not th.is_alive():
            self._feedback_chat(
                lane,
                "[yellow](Nothing to interrupt — turn may have finished.)[/yellow]",
            )
            return
        if not _inject_keyboard_interrupt(th):
            self._feedback_chat(lane, "[yellow]Could not interrupt worker thread.[/yellow]")

    def action_quit(self) -> None:
        self.exit()

    def action_prompt_hist_prev(self) -> None:
        pr = self.query_one("#prompt", TextArea)
        if self.screen.focused is not pr or pr.disabled:
            raise SkipAction()
        lane = self._active_lane
        hist = self._prompt_hist_lines.setdefault(lane, [])
        if not hist:
            raise SkipAction()
        pos = self._prompt_hist_idx.get(lane)
        if pos is None:
            pos = len(hist) - 1
        else:
            pos = max(0, pos - 1)
        self._prompt_hist_idx[lane] = pos
        _set_prompt_text(pr, hist[pos])

    def action_prompt_hist_next(self) -> None:
        pr = self.query_one("#prompt", TextArea)
        if self.screen.focused is not pr or pr.disabled:
            raise SkipAction()
        lane = self._active_lane
        hist = self._prompt_hist_lines.setdefault(lane, [])
        pos = self._prompt_hist_idx.get(lane)
        if pos is None:
            raise SkipAction()
        nxt = pos + 1
        if nxt >= len(hist):
            self._prompt_hist_idx[lane] = None
            _set_prompt_text(pr, "")
        else:
            self._prompt_hist_idx[lane] = nxt
            _set_prompt_text(pr, hist[nxt])

    _SIDEBAR_IDLE_ICON = "○"
    _SIDEBAR_BUSY_ICON = "●"
    _SIDEBAR_MAX_CHARS = 24

    def _sidebar_line_for_lane(self, lane: int, *, label: Optional[str] = None) -> str:
        """One-line sidebar label: status icon + agent name (no model, no wrap)."""
        label = label if label is not None else self._lane_labels[lane]
        busy = lane in self._busy_lanes
        icon = self._SIDEBAR_BUSY_ICON if busy else self._SIDEBAR_IDLE_ICON
        prefix = f"{icon} "
        max_label = max(1, self._SIDEBAR_MAX_CHARS - len(prefix))
        if len(label) > max_label:
            label = label[: max_label - 1] + "…"
        if busy:
            return f"[bold yellow]{icon}[/bold yellow] {escape(label)}"
        return f"[dim]{icon}[/dim] {escape(label)}"

    def _refresh_sidebar_lane(self, lane: int) -> None:
        """Update sidebar option text (agent name + idle/busy icon)."""
        if lane < 0 or lane >= self._n:
            return
        try:
            ol = self.query_one("#agent_list", OptionList)
            ol.replace_option_prompt_at_index(lane, self._sidebar_line_for_lane(lane))
        except Exception:
            pass

    def _mount_lane_widgets(self, idx: int, *, hidden: bool) -> SelectableRichLog:
        # Keep direct refs — query_one on an unmounted Vertical fails in Textual (dynamic /fork).
        thinking_log = SelectableRichLog(
            classes="thinking_log",
            id=f"thinking-{idx}",
            highlight=False,
            markup=True,
            wrap=True,
            auto_scroll=True,
        )
        thinking_panel = Vertical(
            NoSelectStatic("Thinking", classes="thinking_title"),
            thinking_log,
            classes="thinking_panel",
            id=f"thinking-panel-{idx}",
        )
        chat = SelectableRichLog(
            classes="chat_log",
            id=f"chat-{idx}",
            highlight=False,
            markup=True,
            wrap=True,
            auto_scroll=True,
        )
        chat_lane = Vertical(chat, classes="chat_lane", id=f"chat-lane-{idx}")
        lane_cls = "lane-pane hidden" if hidden else "lane-pane"
        lane = Vertical(thinking_panel, chat_lane, classes=lane_cls, id=f"lane-{idx}")
        self._lane_verticals.append(lane)
        self._thinking_logs.append(thinking_log)
        self._chat_logs.append(chat)
        container = self.query_one("#lanes_container", Vertical)
        container.mount(lane)
        return chat

    @staticmethod
    def _thinking_display_text(buf: str) -> str:
        s = buf
        for marker in ("[Thinking]", "[Done thinking]"):
            s = s.replace(marker, "")
        return s.strip()

    def _sync_thinking_log(self, lane: int) -> None:
        log = self._thinking_logs[lane]
        text = self._thinking_display_text(self._thinking_buf.get(lane, ""))
        log.clear()
        if text:
            log.write(Text.from_markup(f"[dim]{escape(text)}[/dim]"))

    def _scroll_chat_to_end(self, lane: int) -> None:
        """Keep chat pinned to latest line after lane layout height changes."""
        chat = self._chat_logs[lane]

        def _scroll() -> None:
            chat.scroll_end(animate=False, immediate=True, x_axis=False)

        self.call_after_refresh(_scroll)

    def _show_thinking_panel(self, lane: int) -> None:
        lane_node = self._lane_verticals[lane]
        if "thinking_open" not in lane_node.classes:
            lane_node.add_class("thinking_open")
            self._scroll_chat_to_end(lane)

    def _hide_thinking_panel(self, lane: int) -> None:
        lane_node = self._lane_verticals[lane]
        was_open = "thinking_open" in lane_node.classes
        lane_node.remove_class("thinking_open")
        self._thinking_logs[lane].clear()
        if was_open:
            self._scroll_chat_to_end(lane)

    def _fork_new_lane(
        self,
        name: str,
        cmds: List[str],
        parent_lane: int,
        *,
        switch_to_new: bool,
    ) -> Optional[int]:
        """
        Create a lane or reuse an existing label. Returns lane index, or ``None`` on failure.
        """
        from agentlib import fork_embedded_session

        nm = (name or "").strip()
        if not nm:
            return None
        matches = self._lanes_matching_name(nm)
        if len(matches) > 1:
            lanes_s = ", ".join(str(i + 1) for i in matches)
            self._feedback_chat(
                parent_lane,
                f"[yellow]Cannot fork {nm!r}: ambiguous name (lanes {lanes_s}).[/yellow]",
            )
            return None
        if len(matches) == 1:
            lane_idx = matches[0]
            ol = self.query_one("#agent_list", OptionList)
            if switch_to_new:
                self._show_lane(lane_idx)
                ol.highlighted = lane_idx
            self._chat_logs[parent_lane].write(
                Text.from_markup(
                    f"[dim]Reused existing lane [bold]{escape(nm)}[/bold] (no duplicate fork).[/dim]\n"
                )
            )
            filtered = [c.strip() for c in cmds if c.strip()]
            if filtered:
                self._execute_lines_chain(lane_idx, filtered)
            self._sync_prompt_enabled()
            return lane_idx

        parent_sess = self._sessions[parent_lane]
        new_idx = self._n
        new_sess = fork_embedded_session(parent_sess, app=self._embed_app)
        new_sess.python_fork_agent = lambda name, cmds=None, li=new_idx: self._fork_lane_hook(
            li, name, cmds, background=False
        )
        new_sess.python_fork_background_agent = lambda name, cmds=None, li=new_idx: self._fork_lane_hook(
            li, name, cmds, background=True
        )

        chat = self._mount_lane_widgets(new_idx, hidden=not switch_to_new)
        ol = self.query_one("#agent_list", OptionList)
        self._lane_labels.append(nm)
        ol.add_option(Option(self._sidebar_line_for_lane(new_idx), id=f"agent-{new_idx}"))
        self._sessions.append(new_sess)
        self._chat_live_buf[new_idx] = ""
        self._chat_stream_open[new_idx] = False
        self._lane_turn_seq[new_idx] = 0
        self._thinking_buf[new_idx] = ""
        self._thinking_follow[new_idx] = False
        self._n += 1

        if switch_to_new:
            hint = (
                f"[bold]{escape(nm)}[/bold] — [dim]forked from lane {parent_lane + 1}[/dim]\n"
                f"[dim]Ctrl+Q quit · /help · /fork …[/dim]"
            )
            parent_note = f"[dim]Fork → [bold]{escape(nm)}[/bold][/dim]\n"
        else:
            hint = (
                f"[bold]{escape(nm)}[/bold] — [dim]background fork from lane {parent_lane + 1}[/dim]\n"
                f"[dim]Select in sidebar when ready · /fork · /fork_background …[/dim]"
            )
            parent_note = f"[dim]Fork (background) → [bold]{escape(nm)}[/bold][/dim]\n"
        chat.write(Text.from_markup(hint))
        self._chat_logs[parent_lane].write(Text.from_markup(parent_note))

        if switch_to_new:
            self._show_lane(new_idx)
            ol.highlighted = new_idx
        self._sync_prompt_enabled()

        filtered = [c.strip() for c in cmds if c.strip()]
        self._prompt_hist_lines[new_idx] = []
        self._prompt_hist_idx.pop(new_idx, None)
        if filtered:
            self._execute_lines_chain(new_idx, filtered)
        return new_idx

    def _handle_fork(self, line: str) -> None:
        parent_lane = self._active_lane
        parsed = parse_fork_command(line)
        if parsed is None:
            self._feedback_chat(
                parent_lane,
                '[yellow]/fork NAME[/yellow] or [yellow]/fork NAME "cmd1,cmd2"[/yellow]',
            )
            return
        name, cmds = parsed
        self._fork_new_lane(name, cmds, parent_lane, switch_to_new=True)

    def _handle_fork_background(self, line: str) -> None:
        parent_lane = self._active_lane
        parsed = parse_fork_background_command(line)
        if parsed is None:
            self._feedback_chat(
                parent_lane,
                '[yellow]/fork_background NAME[/yellow] or '
                '[yellow]/fork_background NAME "cmd1,cmd2"[/yellow]',
            )
            return
        name, cmds = parsed
        self._fork_new_lane(name, cmds, parent_lane, switch_to_new=False)

    def _record_prompt_submission(self, lane: int, text: str) -> None:
        """Append a submitted line to this lane's recall list (dedupe consecutive repeats)."""
        self._prompt_hist_idx[lane] = None
        if not text:
            return
        hist = self._prompt_hist_lines.setdefault(lane, [])
        if hist and hist[-1] == text:
            return
        hist.append(text)
        self._append_startup_lane_submission_to_disk(lane, text)

    def _lanes_matching_name(self, name: str) -> List[int]:
        key = name.casefold().strip()
        if key == "self":
            if 0 <= self._active_lane < len(self._lane_labels):
                return [self._active_lane]
            return []
        return [i for i, lab in enumerate(self._lane_labels) if lab.casefold().strip() == key]

    def _run_on_ui_thread(self, fn: Callable[[], None], *, timeout_s: float = 300.0) -> None:
        """Run ``fn`` on the Textual UI thread and block until it finishes (fork/send hooks)."""
        done = threading.Event()
        err_box: List[BaseException] = []

        def wrapper() -> None:
            try:
                fn()
            except BaseException as e:
                err_box.append(e)
            finally:
                done.set()

        self.call_from_thread(wrapper)
        if not done.wait(timeout=timeout_s):
            raise TimeoutError("Timed out waiting for UI thread")
        if err_box:
            raise err_box[0]

    def _resolve_lane_for_agent(self, agent_name: str) -> tuple[Optional[int], Optional[str]]:
        matches = self._lanes_matching_name(agent_name)
        if not matches:
            return None, f"No agent named {agent_name!r}."
        if len(matches) > 1:
            lanes_s = ", ".join(str(i + 1) for i in matches)
            return None, f"Ambiguous name {agent_name!r} (lanes {lanes_s})."
        return matches[0], None

    def _is_lane_worker_thread(self, lane_idx: int) -> bool:
        from agentlib.tui_busy import is_lane_worker_thread

        return is_lane_worker_thread(self._lane_worker_threads, lane_idx)

    def _wait_until_lane_idle(self, lane_idx: int, poll: float = 0.05) -> None:
        """Block until ``lane_idx`` is not busy (``/turn`` semantics). No-op when already on that lane's worker."""
        from agentlib.tui_busy import wait_until_lane_idle

        def is_busy(ln: int) -> bool:
            busy_box: List[bool] = []
            self.call_from_thread(
                lambda: busy_box.append(ln in self._busy_lanes)
            )
            return bool(busy_box and busy_box[0])

        wait_until_lane_idle(
            lane_idx,
            is_busy=is_busy,
            lane_worker_threads=self._lane_worker_threads,
            poll=poll,
        )

    def _run_turn_for_lane_sync(self, agent_name: str, cmd: str) -> dict:
        """Blocking turn on a lane: wait until idle, run with emit + transcript, return full result."""
        lane_idx, err = self._resolve_lane_for_agent(agent_name)
        if err:
            return {"ok": False, "error": err}
        box: List[dict] = []

        def worker() -> None:
            try:
                box.append(self._python_delegate_bridge(agent_name, cmd.strip()))
            except BaseException as e:
                box.append({"type": "command", "quit": False, "output": f"{type(e).__name__}: {e}"})

        th = threading.Thread(target=worker, name=f"agent-turn-sync-{lane_idx}", daemon=True)
        th.start()
        th.join()
        return {"ok": True, "result": box[0] if box else {"type": "command", "quit": False, "output": ""}}

    def _fork_lane_hook(
        self,
        parent_lane: int,
        name: str,
        commands=None,
        *,
        background: bool = False,
    ) -> dict:
        """Run fork UI on the Textual thread; parent lane is the session's lane index (not sidebar focus)."""
        cmds = [str(c).strip() for c in (commands or []) if str(c).strip()]
        nm = (name or "").strip()
        if not nm:
            return {"type": "fork", "ok": False, "error": "fork requires a non-empty name"}
        box: List[dict] = []

        def ui() -> None:
            try:
                before_n = self._n
                lane_idx = self._fork_new_lane(nm, cmds, parent_lane, switch_to_new=not background)
                if lane_idx is None:
                    box.append({"type": "fork", "ok": False, "error": f"fork {nm!r} failed"})
                elif self._n == before_n:
                    box.append(
                        {
                            "type": "fork",
                            "ok": True,
                            "reused": True,
                            "lane": lane_idx,
                            "label": nm,
                        }
                    )
                else:
                    box.append(
                        {
                            "type": "fork",
                            "ok": True,
                            "lane": lane_idx,
                            "label": nm,
                        }
                    )
            except Exception as e:
                box.append({"type": "fork", "ok": False, "error": str(e)})

        try:
            self._run_on_ui_thread(ui)
        except Exception as e:
            return {"type": "fork", "ok": False, "error": str(e)}
        return box[0] if box else {"type": "fork", "ok": False, "error": "no result"}

    def _python_delegate_bridge(self, agent_name: str, cmd: str) -> dict:
        """Host hook for ``ai(cmd, agent_name)`` inside ``/call_python``."""
        from agentlib.sink import emit_sink_scope, sink_delegate_capture_append

        matches = self._lanes_matching_name((agent_name or "").strip())
        if not matches:
            return {"type": "command", "quit": False, "output": f"No agent named {agent_name!r}."}
        if len(matches) > 1:
            lanes_s = ", ".join(str(i + 1) for i in matches)
            return {
                "type": "command",
                "quit": False,
                "output": f"Ambiguous agent {agent_name!r} (lanes {lanes_s}).",
            }
        lane_idx = matches[0]
        sess = self._sessions[lane_idx]
        cmd_stripped = (cmd or "").strip()
        self._wait_until_lane_idle(lane_idx)
        # Match _run_line: show the user message in this lane's chat before the model runs. Delegate
        # used to call execute_line alone, so the transcript stayed blank until the assistant reply.
        turn_box: List[int] = []

        def prep() -> None:
            turn_box.append(self._prepare_turn_ui(lane_idx, cmd_stripped))

        self.call_from_thread(prep)
        turn_seq = turn_box[0]
        nested_worker = self._is_lane_worker_thread(lane_idx)
        if not nested_worker:
            self.call_from_thread(self._set_lane_busy, lane_idx, True)
        captured: List[str] = []

        def emit_fn(ev, ln=lane_idx, ts=turn_seq):
            self._emit_for(ln, ev, ts)
            sink_delegate_capture_append(ev, captured)

        try:
            with emit_sink_scope(emit_fn):
                res = sess.execute_line(cmd_stripped)
        finally:
            if not nested_worker:
                self.call_from_thread(self._set_lane_busy, lane_idx, False)

        cap = "".join(captured).rstrip()
        if isinstance(res, dict) and cap:
            from agentlib.tools.session_control import merge_command_transcript_output

            if res.get("type") == "command":
                cur = (res.get("output") or "").strip()
                merged = merge_command_transcript_output(cur, cap)
                if merged != cur:
                    res = {**res, "output": merged}
            elif res.get("type") == "turn":
                ans = (res.get("answer") or "").strip()
                merged = merge_command_transcript_output(ans, cap)
                if merged:
                    res = {**res, "answer": merged}

        # Delegate bypasses _run_line → _turn_done; still append final assistant / command
        # output to this lane's transcript (same as _apply_turn_result after a normal turn).
        self.call_from_thread(
            lambda li=lane_idx, r=res, ts=turn_seq: self._apply_turn_result(
                li, r, finalize_busy=False, turn_seq=ts
            )
        )
        return res

    def _python_enqueue_bridge(self, agent_name: str, cmd: str) -> dict:
        """Schedule ``execute_line`` on another lane (main thread); same semantics as ``/send``."""
        box: List[dict] = []

        def ui() -> None:
            try:
                box.append(self._enqueue_turn_for_lane(agent_name.strip(), cmd.strip()))
            except Exception as e:
                box.append({"ok": False, "error": str(e)})

        try:
            self._run_on_ui_thread(ui)
        except Exception as e:
            return {"ok": False, "error": str(e)}
        return box[0] if box else {"ok": False, "error": "no result"}

    def _enqueue_turn_for_lane(self, agent_name: str, cmd: str) -> dict:
        """Must run on the Textual UI thread. Starts ``cmd`` on ``lane`` or appends to its queue."""
        lane_idx, err = self._resolve_lane_for_agent(agent_name)
        if err:
            return {"ok": False, "error": err}
        label = self._lane_labels[lane_idx]
        if lane_idx in self._busy_lanes:
            self._lane_turn_queues.setdefault(lane_idx, []).append(cmd)
            return {"ok": True, "queued": True, "lane": lane_idx, "label": label}
        self._run_line(lane_idx, cmd)
        return {"ok": True, "queued": False, "lane": lane_idx, "label": label}

    def _drain_lane_queue(self, lane: int) -> None:
        q = self._lane_turn_queues.get(lane)
        if not q:
            return
        nxt = q.pop(0)
        if not q:
            self._lane_turn_queues.pop(lane, None)
        self._run_line(lane, nxt)

    def _handle_turn(self, line: str) -> None:
        lane = self._active_lane
        parsed = parse_turn_command(line)
        if parsed is None:
            self._feedback_chat(
                lane,
                "[yellow]Usage:[/yellow] /turn AGENT COMMAND…  ·  "
                '/turn self "cmd1,cmd2,…" (blocking; waits if agent is busy)',
            )
            return
        agent_name, cmds = parsed
        for cmd in cmds:
            r = self._run_turn_for_lane_sync(agent_name, cmd)
            if not r.get("ok"):
                self._feedback_chat(
                    lane,
                    f"[yellow]{escape(str(r.get('error', '/turn failed')))}[/yellow]",
                )
                return
            # Transcript + streaming: _python_delegate_bridge → _apply_turn_result on target lane.

    def _handle_send(self, line: str) -> None:
        lane = self._active_lane
        parsed = parse_send_command(line)
        if parsed is None:
            self._feedback_chat(
                lane,
                "[yellow]Usage:[/yellow] /send AGENT COMMAND…  ·  "
                '/send AGENT "cmd1,cmd2,…" (quotes and commas like /fork)',
            )
            return
        agent_name, cmds = parsed
        for cmd in cmds:
            r = self._enqueue_turn_for_lane(agent_name, cmd)
            if not r.get("ok"):
                self._feedback_chat(
                    lane,
                    f"[yellow]{escape(str(r.get('error', '/send failed')))}[/yellow]",
                )
                return
            lab = escape(str(r.get("label", agent_name)))
            preview = cmd if len(cmd) <= 200 else cmd[:197] + "…"
            preview_esc = escape(preview)
            if r.get("queued"):
                self._feedback_chat(
                    lane,
                    f"[dim]Queued for[/dim] [bold]{lab}[/bold]: [dim]{preview_esc}[/dim]",
                )
            else:
                self._feedback_chat(
                    lane,
                    f"[dim]Started on[/dim] [bold]{lab}[/bold]: [dim]{preview_esc}[/dim]",
                )

    def _python_host_bridge(self, payload: dict) -> dict:
        """Host hook for ``session.host_ctl(...)`` inside ``/call_python`` (main thread)."""
        box: List[dict] = []

        def ui() -> None:
            try:
                box.append(self._host_ctl_dispatch(payload))
            except Exception as e:
                box.append({"ok": False, "error": str(e)})

        try:
            self._run_on_ui_thread(ui)
        except Exception as e:
            return {"ok": False, "error": str(e)}
        return box[0] if box else {"ok": False, "error": "no result"}

    def _host_ctl_dispatch(self, payload: dict) -> dict:
        op = str(payload.get("op") or "")
        raw_arg = payload.get("arg")
        arg_s = str(raw_arg).strip() if raw_arg is not None and str(raw_arg).strip() else ""
        sess = payload.get("session")

        if op == "list_agents":
            lines: List[str] = []
            for i, lab in enumerate(self._lane_labels):
                star = "*" if i == self._active_lane else " "
                lines.append(f"{star} {i + 1}. {lab}")
            return {"ok": True, "text": "\n".join(lines) if lines else "(no agents)"}

        if op == "switch":
            if not arg_s:
                return {"ok": False, "error": "missing agent name"}
            matches = self._lanes_matching_name(arg_s)
            if not matches:
                return {"ok": False, "error": f"No agent named {arg_s!r}"}
            if len(matches) > 1:
                lanes_s = ", ".join(str(i + 1) for i in matches)
                return {"ok": False, "error": f"Ambiguous name {arg_s!r} (lanes {lanes_s})"}
            lane = matches[0]
            ol = self.query_one("#agent_list", OptionList)
            self._active_lane = lane
            self._show_lane(lane)
            ol.highlighted = lane
            self._sync_prompt_enabled()
            return {"ok": True, "text": f"Switched to {self._lane_labels[lane]!r}."}

        if op == "last_answer":
            return self._host_ctl_snapshot(sess, arg_s if arg_s else None, kind="answer")
        if op == "last_question":
            return self._host_ctl_snapshot(sess, arg_s if arg_s else None, kind="question")

        return {"ok": False, "error": f"unknown op {op!r}"}

    def _host_ctl_snapshot(self, source_session, name: Optional[str], *, kind: str) -> dict:
        lane_idx: Optional[int] = None
        if name:
            matches = self._lanes_matching_name(name)
            if not matches:
                return {"ok": False, "error": f"No agent named {name!r}"}
            if len(matches) > 1:
                lanes_s = ", ".join(str(i + 1) for i in matches)
                return {"ok": False, "error": f"Ambiguous name {name!r} (lanes {lanes_s})"}
            lane_idx = matches[0]
        else:
            for i, s in enumerate(self._sessions):
                if s is source_session:
                    lane_idx = i
                    break
            if lane_idx is None:
                return {"ok": False, "error": "Session is not attached to a TUI lane"}
        ts = self._sessions[lane_idx]
        if kind == "answer":
            v = ts.repl_last_assistant_answer
            hint = "(no last assistant answer yet)"
        else:
            v = ts.repl_last_user_query
            hint = "(no last user question yet)"
        text = v.strip() if isinstance(v, str) and v.strip() else hint
        return {"ok": True, "text": text}

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

        st = self._thinking_logs[src]
        dt = self._thinking_logs[dst]
        dt.clear()
        dt.lines.extend(list(st.lines))
        dt._widest_line_width = st._widest_line_width
        dt._start_line = st._start_line
        dt.virtual_size = st.virtual_size
        dt._line_cache.clear()
        dt.refresh()
        if "thinking_open" in self._lane_verticals[src].classes:
            self._lane_verticals[dst].add_class("thinking_open")
        else:
            self._lane_verticals[dst].remove_class("thinking_open")

        self._chat_live_buf[dst] = self._chat_live_buf.get(src, "")
        self._chat_stream_open[dst] = self._chat_stream_open.get(src, False)
        self._lane_turn_seq[dst] = self._lane_turn_seq.get(src, 0)
        if src in self._chat_stream_line_start:
            self._chat_stream_line_start[dst] = self._chat_stream_line_start[src]
        else:
            self._chat_stream_line_start.pop(dst, None)

    def _kill_lane_at(self, k: int) -> None:
        """Remove lane index ``k``; compact indices by swapping last lane into ``k`` when needed."""
        last = self._n - 1
        prev_active = self._active_lane
        ol = self.query_one("#agent_list", OptionList)

        if k != last:
            self._sessions[k] = self._sessions[last]
            self._lane_labels[k] = self._lane_labels[last]
            self._sync_lane_visual_from(last, k)

            ol.replace_option_prompt_at_index(k, self._sidebar_line_for_lane(k))

            self._chat_live_buf[k] = self._chat_live_buf[last]
            self._chat_stream_open[k] = self._chat_stream_open.get(last, False)
            self._lane_turn_seq[k] = self._lane_turn_seq.get(last, 0)
            if last in self._chat_stream_line_start:
                self._chat_stream_line_start[k] = self._chat_stream_line_start[last]
            else:
                self._chat_stream_line_start.pop(k, None)
            self._thinking_buf[k] = self._thinking_buf[last]
            self._thinking_follow[k] = self._thinking_follow[last]

        self._sessions.pop()
        self._lane_labels.pop()
        del self._chat_live_buf[last]
        self._chat_stream_open.pop(last, None)
        self._chat_stream_open.pop(k, None)
        self._lane_turn_seq.pop(last, None)
        self._lane_turn_seq.pop(k, None)
        self._chat_stream_line_start.pop(last, None)
        self._chat_stream_line_start.pop(k, None)
        del self._thinking_buf[last]
        del self._thinking_follow[last]

        self._busy_lanes.discard(last)
        self._busy_lanes.discard(k)

        if k != last:
            tail_hist = self._prompt_hist_lines.pop(last, [])
            self._prompt_hist_lines[k] = list(tail_hist)
        else:
            self._prompt_hist_lines.pop(last, None)
        self._prompt_hist_idx.pop(last, None)
        self._prompt_hist_idx.pop(k, None)

        lane_widget = self._lane_verticals.pop()
        lane_widget.remove()

        self._thinking_logs.pop()
        self._chat_logs.pop()

        ol.remove_option_at_index(last)
        self._lane_turn_queues.pop(k, None)
        self._lane_turn_queues.pop(last, None)
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

        name = parse_kill_command(line)
        if name is None:
            self._feedback_chat(
                fb,
                '[yellow]/kill NAME[/yellow] or [yellow]/kill "Long Name"[/yellow]',
            )
            return
        matches = self._lanes_matching_name(name)
        if not matches:
            self._feedback_chat(
                fb,
                f"[yellow]No agent named[/yellow] [bold]{escape(name)}[/bold]",
            )
            return
        if len(matches) > 1:
            lanes_s = ", ".join(str(i + 1) for i in matches)
            self._feedback_chat(
                fb,
                f"[yellow]Ambiguous name[/yellow] [bold]{escape(name)}[/bold] "
                f"[dim](lanes {lanes_s}); give forks distinct names[/dim]",
            )
            return
        k = matches[0]
        if self._n <= 1:
            self._feedback_chat(fb, "[yellow]Cannot remove the last agent.[/yellow]")
            return

        victim_label = self._lane_labels[k]
        self._kill_lane_at(k)

        ack_lane = self._active_lane
        self._feedback_chat(ack_lane, f"[dim]Killed[/dim] [bold]{escape(victim_label)}[/bold]")

    def _execute_lines_chain(self, lane: int, lines: List[str]) -> None:
        if not lines:
            return
        self._set_lane_busy(lane, True)
        session = self._sessions[lane]

        def worker() -> None:
            try:
                for i, ln in enumerate(lines):
                    last = i == len(lines) - 1
                    turn_box: List[int] = []

                    def prep(l=ln) -> None:
                        turn_box.append(self._prepare_turn_ui(lane, l))

                    self.call_from_thread(prep)
                    turn_seq = turn_box[0]
                    line_emit: Callable[[dict], None] = (
                        lambda ev, ln=lane, ts=turn_seq: self._emit_for(ln, ev, ts)
                    )
                    try:
                        res = session.execute_line(ln, emit=line_emit)
                    except KeyboardInterrupt:
                        self.call_from_thread(self._turn_cancelled, lane)
                        return
                    self.call_from_thread(
                        lambda r=res, ts=turn_seq: self._apply_turn_result(
                            lane, r, finalize_busy=last, turn_seq=ts
                        )
                    )
            except BaseException:
                tb = traceback.format_exc()
                self.call_from_thread(self._turn_error, lane, tb)
            finally:

                def _clear_worker_ref(ln: int = lane) -> None:
                    self._lane_worker_threads.pop(ln, None)

                self.call_from_thread(_clear_worker_ref)

        th = threading.Thread(target=worker, name=f"agent-chain-{lane}", daemon=True)
        self._lane_worker_threads[lane] = th
        th.start()

    def _turn_seq_current(self, lane: int) -> int:
        return int(self._lane_turn_seq.get(lane, 0))

    def _turn_seq_is_current(self, lane: int, turn_seq: int) -> bool:
        return self._turn_seq_current(lane) == int(turn_seq)

    def _commit_chat_live_answer(self, lane: int) -> None:
        """Keep the visible streamed reply in the log; end live-edit mode for a new generation."""
        if not self._chat_stream_open.get(lane):
            return
        chat = self._chat_logs[lane]
        if self._chat_live_buf.get(lane, "").strip():
            chat.write(Text("\n"), scroll_end=True)
        self._reset_chat_live_answer(lane)

    def _discard_chat_live_answer(self, lane: int) -> None:
        """Drop the in-progress assistant block from the log and clear stream tracking."""
        if self._chat_stream_open.get(lane):
            self._chat_truncate_live_block(lane)
        self._reset_chat_live_answer(lane)

    def _reset_chat_live_answer(self, lane: int) -> None:
        self._chat_live_buf[lane] = ""
        self._chat_stream_open[lane] = False
        self._chat_stream_line_start.pop(lane, None)

    def _chat_truncate_live_block(self, lane: int) -> None:
        """Remove all RichLog lines belonging to the current in-progress assistant reply."""
        chat = self._chat_logs[lane]
        start = self._chat_stream_line_start.get(lane)
        if start is None or start >= len(chat.lines):
            return
        del chat.lines[start:]
        chat._line_cache.clear()
        chat.refresh()

    def _show_draft_enabled(self, lane: int) -> bool:
        if 0 <= lane < len(self._sessions):
            return self._sessions[lane].settings.get_bool(("agent", "show_draft"), False)
        return False

    def _chat_rewrite_live_draft(self, lane: int) -> None:
        buf = self._chat_live_buf.get(lane, "")
        chat = self._chat_logs[lane]
        if self._chat_stream_open.get(lane):
            self._chat_truncate_live_block(lane)
        else:
            self._chat_stream_line_start[lane] = len(chat.lines)
        self._chat_stream_open[lane] = True
        if self._show_draft_enabled(lane):
            body = Text.from_markup(f"[bold dim]Draft[/bold dim]\n{escape(buf)}")
        else:
            body = Text.from_markup(f"[bold cyan]Assistant[/bold cyan]\n{escape(buf)}")
        chat.write(body, scroll_end=True)
        chat.refresh()

    def _streamed_answer_matches_final(self, lane: int, body: str) -> bool:
        if not self._chat_stream_open.get(lane):
            return False
        live = self._chat_live_buf.get(lane, "").strip()
        return bool(live) and live == body

    def _write_final_answer_block(self, lane: int, text: str) -> None:
        body = (text or "").strip()
        if not body:
            return
        if not self._show_draft_enabled(lane) and self._streamed_answer_matches_final(lane, body):
            self._chat_close_live_answer(lane)
            return
        self._chat_close_live_answer(lane)
        chat = self._chat_logs[lane]
        chat.write(
            Text.from_markup(f"[bold green]Final[/bold green]\n{escape(body)}\n"),
            scroll_end=True,
        )

    def _chat_set_live_answer_snapshot(self, lane: int, text: str) -> None:
        prev = self._chat_live_buf.get(lane, "")
        if text == prev:
            return
        self._hide_thinking_panel(lane)
        if self._chat_stream_open.get(lane) and prev and not text.startswith(prev):
            from agentlib.llm.streaming import merge_visible_answer_text

            merged = merge_visible_answer_text(prev, text)
            if merged.startswith(prev) or merged == prev:
                text = merged
            else:
                self._commit_chat_live_answer(lane)
        self._chat_live_buf[lane] = text
        self._chat_rewrite_live_draft(lane)

    def _chat_append_answer_delta(self, lane: int, delta: str) -> None:
        if not delta:
            return
        self._hide_thinking_panel(lane)
        from agentlib.llm.streaming import merge_visible_answer_text

        buf = self._chat_live_buf.get(lane, "")
        merged = merge_visible_answer_text(buf, delta)
        if merged == buf:
            return
        self._chat_live_buf[lane] = merged
        self._chat_rewrite_live_draft(lane)

    def _chat_close_live_answer(self, lane: int) -> None:
        """End a streamed reply without re-printing it (already shown via partial updates)."""
        if not self._chat_stream_open.get(lane):
            return
        chat = self._chat_logs[lane]
        if self._chat_live_buf.get(lane, "").strip():
            chat.write(Text("\n"), scroll_end=True)
        self._reset_chat_live_answer(lane)

    def _reset_lane_gen_rate_tracker(self, lane: int) -> None:
        """Drop in-flight sampling state; keep last measured tok/s for display."""
        self._lane_rate_tracker.pop(lane, None)
        self._lane_gen_rate_paint_at.pop(lane, None)

    def _maybe_paint_header_gen_rate(self, lane: int, *, force: bool = False) -> None:
        """Sample tok/s while busy; refresh header icon at most once per second."""
        if lane != self._active_lane:
            return
        measuring = lane not in self._lane_gen_tok_s
        if lane in self._busy_lanes:
            now = time.monotonic()
            last = self._lane_gen_rate_paint_at.get(lane, 0.0)
            if force or (now - last) >= self._header_gen_rate_interval_s:
                self._lane_gen_rate_paint_at[lane] = now
                tr = self._lane_rate_tracker.get(lane)
                if tr is not None:
                    rate = tr.sample_interval(
                        min_elapsed=0.2 if force else self._header_gen_rate_interval_s * 0.9
                    )
                    if rate is not None:
                        self._lane_gen_tok_s[lane] = rate
                measuring = lane not in self._lane_gen_tok_s
            elif not force:
                return
        self._refresh_header_gen_rate(lane, measuring=measuring)

    def _track_gen_rate_delta(self, lane: int, delta_text: str) -> None:
        """Accumulate streamed answer tokens (rate sampled on header refresh cadence)."""
        if not delta_text or lane not in self._busy_lanes:
            return
        buf = self._chat_live_buf.get(lane, "")
        if buf.endswith(delta_text):
            return
        tr = self._lane_rate_tracker.get(lane)
        if tr is None:
            tr = GenRateTracker()
            self._lane_rate_tracker[lane] = tr
        tr.add_tokens(estimate_tokens_from_text(delta_text))
        self._maybe_paint_header_gen_rate(lane)

    def _refresh_header_gen_rate(self, lane: Optional[int] = None, *, measuring: bool = False) -> None:
        """Show tok/s in the header icon slot (top-left); title stays ``Agent TUI``."""
        ln = self._active_lane if lane is None else lane
        if ln == self._active_lane:
            self.title = "Agent TUI"
            self.sub_title = ""
        if ln != self._active_lane:
            return
        try:
            header = self.query_one("#app_header", Header)
        except Exception:
            return
        if ln in self._lane_gen_tok_s:
            rate = self._lane_gen_tok_s[ln]
            if rate >= 100:
                header.icon = f"{rate:.0f}/s"
            else:
                header.icon = f"{rate:.1f}/s"
        elif measuring and ln in self._busy_lanes:
            header.icon = "…/s"
        else:
            header.icon = ""

    def _prepare_turn_ui(self, lane: int, line: str) -> int:
        if self._chat_stream_open.get(lane):
            self._commit_chat_live_answer(lane)
        else:
            self._reset_chat_live_answer(lane)
        seq = self._turn_seq_current(lane) + 1
        self._lane_turn_seq[lane] = seq
        chat = self._chat_logs[lane]
        chat.write(Text.from_markup(f"[bold green]You[/bold green]\n{escape(line)}\n"))
        self._reset_lane_gen_rate_tracker(lane)
        self._thinking_buf[lane] = ""
        self._thinking_follow[lane] = False
        self._hide_thinking_panel(lane)
        return seq

    def _show_lane(self, index: int) -> None:
        index = max(0, min(index, self._n - 1))
        self._active_lane = index
        for i in range(self._n):
            lane_node = self._lane_verticals[i]
            if i != index:
                lane_node.add_class("hidden")
            else:
                lane_node.remove_class("hidden")
        self._refresh_header_gen_rate()

    def _feedback_chat(self, lane: int, markup: str) -> None:
        """Transient slash / REPL command text in the chat log (same band as ``/show`` via sink)."""
        self._chat_logs[lane].write(Text.from_markup(markup))

    @on(OptionList.OptionSelected, "#agent_list")
    def agent_selected(self, event: OptionList.OptionSelected) -> None:
        self._show_lane(event.option_index)
        self._prompt_hist_idx[self._active_lane] = None
        pr = self.query_one("#prompt", TextArea)
        _set_prompt_text(pr, "")
        self._sync_prompt_enabled()

    def _emit_for(self, lane: int, ev: dict, turn_seq: Optional[int] = None) -> None:
        payload = dict(ev)
        self.call_from_thread(self._dispatch_emit, lane, payload, turn_seq)

    def _dispatch_emit(self, lane: int, ev: dict, turn_seq: Optional[int] = None) -> None:
        if turn_seq is not None and not self._turn_seq_is_current(lane, turn_seq):
            return
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

        chat = self._chat_logs[lane]

        if t == "thinking":
            if "[Thinking]" in text:
                self._thinking_follow[lane] = True
                self._thinking_buf[lane] = ""
                self._show_thinking_panel(lane)
                self._thinking_logs[lane].clear()
            self._thinking_buf[lane] += text
            rate_delta = text
            for marker in ("[Thinking]", "[Done thinking]"):
                rate_delta = rate_delta.replace(marker, "")
            if rate_delta:
                self._track_gen_rate_delta(lane, rate_delta)
            if "[Done thinking]" in text:
                self._thinking_follow[lane] = False
                self._sync_thinking_log(lane)
                return
            if self._thinking_display_text(self._thinking_buf[lane]):
                self._show_thinking_panel(lane)
                self._sync_thinking_log(lane)
            return

        if t in ("progress", "warning", "stderr", "debug", "error"):
            style = "yellow" if t == "warning" else "red" if t == "error" else "dim"
            chat.write(Text.from_markup(f"[{style}]{escape(text)}[/{style}]"))
            return

        if t == "final_answer":
            self._hide_thinking_panel(lane)
            self._write_final_answer_block(lane, text)
            return

        if t == "answer_reset":
            self._hide_thinking_panel(lane)
            self._commit_chat_live_answer(lane)
            return

        if t == "answer_commit":
            self._hide_thinking_panel(lane)
            self._commit_chat_live_answer(lane)
            return

        if t == "answer":
            self._hide_thinking_panel(lane)
            if partial:
                if ev.get("full_snapshot"):
                    prev = self._chat_live_buf.get(lane, "")
                    if text.startswith(prev):
                        self._track_gen_rate_delta(lane, text[len(prev) :])
                    elif text != prev:
                        self._track_gen_rate_delta(lane, text)
                    self._chat_set_live_answer_snapshot(lane, text)
                else:
                    self._track_gen_rate_delta(lane, text)
                    self._chat_append_answer_delta(lane, text)
                return
            self._write_final_answer_block(lane, text)
            return

        if t == "output":
            if partial:
                return
            if _is_activity_output_line(text):
                chat.write(Text(escape(text)))
            else:
                chat.write(Text.from_markup(f"[dim]{escape(text)}[/dim]"))
            return

        chat.write(Text.from_markup(f"[magenta]{t}[/magenta] {escape(text)}"))

    def _sync_prompt_enabled(self) -> None:
        pr = self.query_one("#prompt", TextArea)
        blocked = self._active_lane in self._busy_lanes
        pr.disabled = blocked
        if not blocked:
            pr.focus()

    def _set_lane_busy(self, lane: int, busy: bool) -> None:
        if busy:
            self._busy_lanes.add(lane)
            if lane == self._active_lane:
                self._lane_gen_rate_paint_at.pop(lane, None)
                self._maybe_paint_header_gen_rate(lane, force=True)
        else:
            if lane == self._active_lane:
                self._maybe_paint_header_gen_rate(lane, force=True)
            self._busy_lanes.discard(lane)
            if lane == self._active_lane:
                self._refresh_header_gen_rate(lane)
        self._refresh_sidebar_lane(lane)
        self._sync_prompt_enabled()

    def action_submit_prompt_message(self) -> None:
        """Send the prompt (invoked when Enter is pressed in ``PromptTextArea``)."""
        pr = self.query_one("#prompt", TextArea)
        if self.screen.focused is not pr or pr.disabled:
            raise SkipAction()
        line = (pr.text or "").strip()
        _set_prompt_text(pr, "")
        lane = self._active_lane
        if not line or lane >= len(self._sessions):
            return
        if self._active_lane in self._busy_lanes:
            return
        self._record_prompt_submission(lane, line)
        if line.startswith("/send"):
            self._handle_send(line)
            return
        if line.startswith("/turn"):
            self._handle_turn(line)
            return
        if line.startswith("/kill"):
            self._handle_kill(line)
            return
        if line.startswith("/fork_background"):
            self._handle_fork_background(line)
            return
        if line.startswith("/fork"):
            self._handle_fork(line)
            return
        self._run_line(lane, line)

    def _run_line(self, lane: int, line: str) -> None:
        turn_seq = self._prepare_turn_ui(lane, line)
        self._set_lane_busy(lane, True)

        session = self._sessions[lane]

        def worker() -> None:
            emit: Callable[[dict], None] = lambda ev, ln=lane, ts=turn_seq: self._emit_for(ln, ev, ts)
            try:
                res = session.execute_line(line, emit=emit)
                self.call_from_thread(self._turn_done, lane, res, turn_seq)
            except KeyboardInterrupt:
                self.call_from_thread(self._turn_cancelled, lane)
            except BaseException:
                tb = traceback.format_exc()
                self.call_from_thread(self._turn_error, lane, tb)
            finally:

                def _clear_worker_ref(ln: int = lane) -> None:
                    self._lane_worker_threads.pop(ln, None)

                self.call_from_thread(_clear_worker_ref)

        th = threading.Thread(target=worker, name=f"agent-turn-{lane}", daemon=True)
        self._lane_worker_threads[lane] = th
        th.start()

    def _turn_error(self, lane: int, tb: str) -> None:
        self._set_lane_busy(lane, False)
        self._chat_logs[lane].write(Text.from_markup(f"[bold red]Turn error[/bold red]\n{escape(tb)}"))
        self._reset_chat_live_answer(lane)
        self._thinking_buf[lane] = ""
        self._thinking_follow[lane] = False
        self._hide_thinking_panel(lane)

    def _turn_cancelled(self, lane: int) -> None:
        """Worker raised KeyboardInterrupt (user confirmed cancel on the interrupt prompt)."""
        self._lane_turn_queues.pop(lane, None)
        self._feedback_chat(lane, "[yellow][Cancelled][/yellow]")
        self._reset_chat_live_answer(lane)
        self._thinking_buf[lane] = ""
        self._thinking_follow[lane] = False
        self._hide_thinking_panel(lane)
        self._set_lane_busy(lane, False)

    def _turn_done(self, lane: int, res: dict, turn_seq: int) -> None:
        self._apply_turn_result(lane, res, finalize_busy=True, turn_seq=turn_seq)

    def _apply_turn_result(
        self, lane: int, res: dict, *, finalize_busy: bool, turn_seq: int
    ) -> None:
        if not self._turn_seq_is_current(lane, turn_seq):
            return
        try:
            if res.get("quit"):
                self.exit()
                return
            chat = self._chat_logs[lane]

            if res.get("type") == "command":
                out = res.get("output") or ""
                if str(out).strip():
                    chat.write(Text.from_markup(f"[yellow]{escape(str(out))}[/yellow]"))
                pre = res.get("prefill_prompt")
                if (
                    isinstance(pre, str)
                    and pre
                    and lane == self._active_lane
                ):
                    pr = self.query_one("#prompt", TextArea)
                    _set_prompt_text(pr, pre)
                    pr.focus()
                return

            if res.get("type") == "turn":
                ans = res.get("answer")
                if isinstance(ans, str) and ans.strip():
                    self._write_final_answer_block(lane, ans)
        finally:
            if not self._turn_seq_is_current(lane, turn_seq):
                return
            self._reset_chat_live_answer(lane)
            self._thinking_buf[lane] = ""
            self._thinking_follow[lane] = False
            self._hide_thinking_panel(lane)
            if finalize_busy:
                self._set_lane_busy(lane, False)
                self._drain_lane_queue(lane)
            if not res.get("quit") and 0 <= lane < self._n:
                self._refresh_sidebar_lane(lane)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Multi-agent TUI (embedded sessions + streaming emit).")
    p.add_argument("-v", "--verbose", action="count", default=0)
    p.add_argument(
        "--debug_log",
        "--debug-log",
        dest="debug_llm_log",
        metavar="FILE",
        default=None,
        help="Append full LLM request JSON (verbose-3 style) to FILE; no extra TUI output.",
    )
    p.add_argument(
        "--agent",
        action="append",
        dest="agents",
        metavar="LABEL[:MODEL]",
        help="Agent slot (repeat). Examples: Planner:llama3.2:latest or Research",
    )
    args = p.parse_args(argv)
    AgentTuiApp(
        verbose=int(args.verbose),
        agent_specs=args.agents,
        debug_llm_log_path=args.debug_llm_log,
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
