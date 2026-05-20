"""Textual widgets for agent_tui: mouse selection confined to log panes."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

from rich.segment import Segment
from rich.style import Style

try:
    from textual import events
    from textual.screen import Screen
    from textual.strip import Strip
    from textual.widgets import RichLog, Static
    from textual.widgets import TextArea as TextualTextArea
except ImportError:
    RichLog = object  # type: ignore[misc, assignment]
    Screen = object  # type: ignore[misc, assignment]
    Static = object  # type: ignore[misc, assignment]
    TextualTextArea = object  # type: ignore[misc, assignment]

PromptTextArea = TextualTextArea

Cell = Tuple[int, int]

# Textual / xterm: left press is usually 1; some paths use 0.
_LEFT_MOUSE_BUTTONS = frozenset({0, 1})

_COPY_SELECTION_KEYS = frozenset(
    {
        "super+c",
        "meta+c",
        "hyper+c",
        "ctrl+shift+c",
        "super+shift+c",
        "meta+shift+c",
    }
)


def is_mac_copy_key(key: str) -> bool:
    """True for Cmd+C-style keys (meta/super+c) that some terminals send."""
    if key in _COPY_SELECTION_KEYS:
        return True
    parts = key.split("+")
    return (
        len(parts) >= 2
        and parts[-1] == "c"
        and any(p in ("super", "meta", "hyper") for p in parts[:-1])
        and "ctrl" not in parts
    )


def is_copy_selection_key(key: str) -> bool:
    """True for any key chord we treat as 'copy log selection'."""
    return key == "ctrl+shift+c" or is_mac_copy_key(key)

MouseEvent = Union["events.MouseDown", "events.MouseMove", "events.MouseUp"]


def extract_plain_selection(
    lines: List[str], start: Cell, end: Cell
) -> str:
    """Return plain text between two (line, column) cells in ``lines``."""
    if not lines:
        return ""
    (y1, x1), (y2, x2) = start, end
    if (y2, x2) < (y1, x1):
        y1, x1, y2, x2 = y2, x2, y1, x1
    y1 = max(0, min(y1, len(lines) - 1))
    y2 = max(0, min(y2, len(lines) - 1))
    if y1 == y2:
        line = lines[y1]
        x1 = max(0, min(x1, len(line)))
        x2 = max(0, min(x2, len(line)))
        if x2 < x1:
            x1, x2 = x2, x1
        return line[x1:x2]
    parts: List[str] = []
    parts.append(lines[y1][max(0, min(x1, len(lines[y1]))):])
    for row in range(y1 + 1, y2):
        parts.append(lines[row])
    end_line = lines[y2]
    parts.append(end_line[: max(0, min(x2, len(end_line)))])
    return "\n".join(parts)


def selection_span_on_line(
    line_index: int,
    scroll_x: int,
    viewport_width: int,
    start: Cell,
    end: Cell,
) -> Optional[Tuple[int, int]]:
    """Viewport column span to highlight on one visible line, or None."""
    (y1, x1), (y2, x2) = start, end
    if (y2, x2) < (y1, x1):
        y1, x1, y2, x2 = y2, x2, y1, x1
    if line_index < y1 or line_index > y2:
        return None
    if y1 == y2:
        a, b = x1, x2
    elif line_index == y1:
        a, b = x1, 10**9
    elif line_index == y2:
        a, b = 0, x2
    else:
        a, b = 0, 10**9
    left = max(0, a - scroll_x)
    right = min(viewport_width, b - scroll_x)
    if right <= left:
        return None
    return left, right


class NoSelectStatic(Static):
    ALLOW_SELECT = False


class SelectableRichLog(RichLog):
    """
    RichLog with drag-to-select inside this pane only.

    Selection is tracked locally and painted in ``render_line``. Mouse routing
    is handled in ``AgentScreen`` so events are not lost to RichLog hit-testing.
    """

    ALLOW_SELECT = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._plain_lines: List[str] = []
        self._sel_start: Optional[Cell] = None
        self._sel_end: Optional[Cell] = None
        self._drag_select = False

    def _rebuild_plain_lines(self) -> None:
        self._plain_lines = [self.lines[i].text for i in range(len(self.lines))]

    def write(self, content, *args, **kwargs):
        result = super().write(content, *args, **kwargs)
        self._rebuild_plain_lines()
        return result

    def clear(self):
        self._clear_mouse_selection()
        self._plain_lines.clear()
        return super().clear()

    def _clear_mouse_selection(self) -> None:
        self._sel_start = None
        self._sel_end = None
        self._drag_select = False
        self.refresh()

    def has_text_selection(self) -> bool:
        return (
            self._sel_start is not None
            and self._sel_end is not None
            and self._sel_start != self._sel_end
        )

    def selected_text(self) -> str:
        if self._sel_start is None or self._sel_end is None:
            return ""
        return extract_plain_selection(self._plain_lines, self._sel_start, self._sel_end)

    def copy_selection_to_clipboard(self) -> bool:
        """Copy the current in-pane selection via ``pbcopy`` / platform CLI."""
        text = self.selected_text()
        if not text.strip():
            return False
        try:
            from agentlib.clipboard_io import clipboard_write_text

            clipboard_write_text(text)
        except Exception:
            return False
        return True

    @staticmethod
    def _is_left_button(event: events.MouseEvent) -> bool:
        return event.button in _LEFT_MOUSE_BUTTONS

    def _cell_at(self, event: events.MouseEvent) -> Cell:
        """Map widget-local mouse coords (from AgentScreen) to content line/column."""
        scroll_x, scroll_y = self.scroll_offset
        line_y = int(scroll_y) + max(0, event.y)
        col = int(scroll_x) + max(0, event.x)
        if not self._plain_lines:
            return 0, 0
        line_y = max(0, min(line_y, len(self._plain_lines) - 1))
        col = max(0, min(col, len(self._plain_lines[line_y])))
        return line_y, col

    def process_mouse(self, event: MouseEvent) -> None:
        """Update selection from a widget-local mouse event (called by AgentScreen)."""
        if isinstance(event, events.MouseDown):
            if not self._is_left_button(event):
                return
            try:
                self.screen.clear_selection()
            except Exception:
                pass
            self._drag_select = True
            self.capture_mouse()
            cell = self._cell_at(event)
            self._sel_start = cell
            self._sel_end = cell
            self.refresh()
        elif isinstance(event, events.MouseMove):
            if not self._drag_select:
                return
            self._sel_end = self._cell_at(event)
            self.refresh()
        elif isinstance(event, events.MouseUp):
            if not self._drag_select:
                return
            self._drag_select = False
            self._sel_end = self._cell_at(event)
            if self.app.mouse_captured is self:
                self.release_mouse()
            self.refresh()
            # macOS Terminal often intercepts Cmd+C; copy when the drag finishes.
            if self.has_text_selection():
                self.copy_selection_to_clipboard()

    async def _on_key(self, event: events.Key) -> None:
        if is_copy_selection_key(event.key) and self.has_text_selection():
            if self.copy_selection_to_clipboard():
                event.stop()
                event.prevent_default()
                return
        await super()._on_key(event)

    def _selection_style(self) -> Style:
        """Readable selection colors that replace dim/reverse on markup segments."""
        try:
            base = self.screen.get_component_rich_style("screen--selection")
            bgcolor = base.bgcolor
            color = base.color
        except Exception:
            bgcolor = None
            color = None
        if bgcolor is None or color is None or bgcolor == color:
            bgcolor = "#4a6fa5"
            color = "#f0f0f0"
        return Style(bgcolor=bgcolor, color=color, dim=False, reverse=False, bold=False)

    @staticmethod
    def _paint_selection(strip: Strip, style: Style) -> Strip:
        """Paint selection with a flat style (do not merge/reverse underlying dim markup)."""
        segments = [Segment(seg.text, style) for seg in strip]
        return Strip(segments, strip.cell_length)

    def _highlight_strip(self, strip: Strip, line_index: int) -> Strip:
        if self._sel_start is None or self._sel_end is None:
            return strip
        scroll_x, _ = self.scroll_offset
        span = selection_span_on_line(
            line_index,
            scroll_x,
            self.scrollable_content_region.width,
            self._sel_start,
            self._sel_end,
        )
        if span is None:
            return strip
        left, right = span
        width = strip.cell_length
        left = max(0, min(left, width))
        right = max(0, min(right, width))
        if right <= left:
            return strip
        sel_style = self._selection_style()
        parts: List[Strip] = []
        if left > 0:
            parts.append(strip.crop(0, left))
        parts.append(self._paint_selection(strip.crop(left, right), sel_style))
        if right < width:
            parts.append(strip.crop(right, width))
        return Strip.join(parts)

    def render_line(self, y: int) -> Strip:
        scroll_x, scroll_y = self.scroll_offset
        line_index = scroll_y + y
        strip = super().render_line(y)
        return self._highlight_strip(strip, line_index)


class AgentScreen(Screen):
    """Route log mouse selection here; built-in selection only in the prompt TextArea."""

    def _widget_uses_builtin_selection(self, widget) -> bool:
        return isinstance(widget, TextualTextArea)

    @staticmethod
    def _local_mouse_event(log: SelectableRichLog, event: events.MouseEvent) -> events.MouseEvent:
        region = log.screen.find_widget(log).region
        return event._apply_offset(-region.x, -region.y)

    def _log_for_mouse_event(self, event: events.MouseEvent) -> Optional[SelectableRichLog]:
        cap = self.app.mouse_captured
        if isinstance(cap, SelectableRichLog):
            return cap
        try:
            widget, _ = self.get_widget_at(event.screen_x, event.screen_y)
        except Exception:
            return None
        return widget if isinstance(widget, SelectableRichLog) else None

    def _forward_event(self, event: events.Event) -> None:
        if isinstance(event, events.MouseEvent):
            log = self._log_for_mouse_event(event)
            if log is not None:
                if isinstance(event, events.MouseDown):
                    self.clear_selection()
                    self._selecting = False
                local = self._local_mouse_event(log, event)
                log.process_mouse(local)
                if isinstance(event, events.MouseUp):
                    self._mouse_down_offset = None
                    self._selecting = False
                    self.post_message(events.TextSelected())
                self.update_pointer_shape()
                return

        if isinstance(event, events.MouseDown) and not self.app.mouse_captured:
            try:
                widget, _ = self.get_widget_at(event.screen_x, event.screen_y)
            except Exception:
                widget = None
            if not self._widget_uses_builtin_selection(widget):
                prev = getattr(widget, "ALLOW_SELECT", None) if widget is not None else None
                if widget is not None:
                    widget.ALLOW_SELECT = False
                try:
                    super()._forward_event(event)
                finally:
                    if widget is not None and prev is not None:
                        widget.ALLOW_SELECT = prev
                return

        super()._forward_event(event)
