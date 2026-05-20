"""Mouse selection helpers for agent_tui log panes."""

import asyncio

from textual.app import App, ComposeResult
from textual.events import MouseDown, MouseMove, MouseUp
from textual.widgets import Footer

from agentlib.tui_widgets import (
    SelectableRichLog,
    AgentScreen,
    extract_plain_selection,
    _LEFT_MOUSE_BUTTONS,
    is_mac_copy_key,
)
from textual.binding import Binding


def test_extract_plain_selection_single_line():
    lines = ["hello world"]
    assert extract_plain_selection(lines, (0, 0), (0, 5)) == "hello"
    assert extract_plain_selection(lines, (0, 6), (0, 11)) == "world"


def test_extract_plain_selection_multiline():
    lines = ["abc", "defghi", "jk"]
    assert extract_plain_selection(lines, (0, 1), (2, 1)) == "bc\ndefghi\nj"


def test_left_mouse_buttons_include_zero_and_one():
    assert 0 in _LEFT_MOUSE_BUTTONS and 1 in _LEFT_MOUSE_BUTTONS


def test_mac_copy_key_detection():
    assert is_mac_copy_key("meta+c")
    assert is_mac_copy_key("super+c")
    assert not is_mac_copy_key("ctrl+c")


def test_copy_binding_includes_meta_c():
    keys = [b.key for b in Binding.make_bindings([Binding("ctrl+shift+c,super+c,meta+c", "copy_mouse_selection")])]
    assert "meta+c" in keys


def test_mouse_up_copies_to_clipboard(monkeypatch):
    copied: list[str] = []

    def _write(text: str) -> None:
        copied.append(text)

    monkeypatch.setattr("agentlib.clipboard_io.clipboard_write_text", _write)

    async def run():
        app = _SelApp()
        async with app.run_test(size=(80, 24)) as pilot:
            log = app.query_one("#log", SelectableRichLog)
            log.write("hello world\nsecond line")
            log._sel_start = (0, 0)
            log._sel_end = (0, 5)
            log._drag_select = True
            up = MouseUp(log, 5, 0, 0, 0, 1, False, False, False, screen_x=5, screen_y=5)
            log.process_mouse(up)
            assert copied == ["hello"]

    asyncio.run(run())


class _SelApp(App):
    def get_default_screen(self):
        return AgentScreen()

    def compose(self):
        yield SelectableRichLog(id="log", markup=True)
        yield Footer()

    async def on_mount(self):
        self.query_one(SelectableRichLog).write("hello world\nsecond line")


def test_selectable_rich_log_accepts_button_zero():
    async def run():
        app = _SelApp()
        async with app.run_test(size=(80, 24)) as pilot:
            log = app.query_one("#log", SelectableRichLog)
            down = MouseDown(
                log, 2, 1, 0, 0, 0, False, False, False, screen_x=5, screen_y=5
            )
            assert log._is_left_button(down)
            log.process_mouse(down)
            assert log._drag_select
            move = MouseMove(
                log, 12, 1, 10, 0, 0, False, False, False, screen_x=15, screen_y=5
            )
            log.process_mouse(move)
            up = MouseUp(
                log, 12, 1, 0, 0, 0, False, False, False, screen_x=15, screen_y=5
            )
            log.process_mouse(up)
            assert log._sel_start is not None
            assert log.selected_text().strip()

    asyncio.run(run())


def test_copy_mouse_selection_uses_os_clipboard(monkeypatch):
    copied: list[str] = []

    def _write(text: str) -> None:
        copied.append(text)

    monkeypatch.setattr("agentlib.clipboard_io.clipboard_write_text", _write)

    async def run():
        from agent_tui import AgentTuiApp

        async with AgentTuiApp(verbose=0, agent_specs=["A"]).run_test(size=(80, 24)) as pilot:
            log = pilot.app.query_one("#chat-0", SelectableRichLog)
            log.write("hello world")
            line_idx = next(i for i, t in enumerate(log._plain_lines) if t.startswith("hello"))
            log._sel_start = (line_idx, 0)
            log._sel_end = (line_idx, 5)
            pilot.app.action_copy_mouse_selection()
            assert copied == ["hello"]

    asyncio.run(run())
