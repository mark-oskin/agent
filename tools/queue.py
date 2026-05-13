"""
Global in-memory named lists (FIFO) for agents and the REPL.

**LLM tools** (enable the ``queue`` toolset): ``list_add``, ``list_remove``, ``list_peek``,
``list_length``, ``list_clear``, ``list_names``.

**REPL** (after ``/load extensions/queue_control.py``): ``/queue show lists``,
``/queue <name> add …``, ``remove``, ``peek``, ``length``, ``clear``, ``save <file>``, ``load <file>``.

``list_remove`` and ``list_peek`` return the literal ``<empty>`` when the list is missing or empty.
"""

from __future__ import annotations

import json
from collections import deque
from threading import RLock
from typing import Dict, List

from agentlib.coercion import scalar_to_str

_EMPTY = "<empty>"
_LOCK = RLock()
_STORE: Dict[str, deque[str]] = {}


def reset_queues_for_testing() -> None:
    """Clear all lists (for tests only)."""
    with _LOCK:
        _STORE.clear()


def _norm_listname(raw: object) -> str:
    return scalar_to_str(raw, "").strip()


def queue_add(name: str, data: str) -> str:
    name = _norm_listname(name)
    data = scalar_to_str(data, "")
    if not name:
        return "list_add error: list name is empty."
    if not data:
        return "list_add error: data is empty."
    with _LOCK:
        d = _STORE.setdefault(name, deque())
        d.append(data)
        n = len(d)
    return f"list_add: appended to {name!r} (length {n})."


def queue_remove(name: str) -> str:
    name = _norm_listname(name)
    if not name:
        return "queue_remove error: list name is empty."
    with _LOCK:
        d = _STORE.get(name)
        if not d:
            return _EMPTY
        item = d.popleft()
    return item


def queue_peek(name: str) -> str:
    name = _norm_listname(name)
    if not name:
        return "queue_peek error: list name is empty."
    with _LOCK:
        d = _STORE.get(name)
        if not d:
            return _EMPTY
        item = d[0]
    return item


def queue_length(name: str) -> str:
    name = _norm_listname(name)
    if not name:
        return "queue_length error: list name is empty."
    with _LOCK:
        d = _STORE.get(name)
        n = len(d) if d else 0
    return str(n)


def queue_clear(name: str) -> str:
    name = _norm_listname(name)
    if not name:
        return "queue_clear error: list name is empty."
    with _LOCK:
        had = name in _STORE and len(_STORE[name]) > 0
        _STORE[name] = deque()
    return f"queue_clear: list {name!r} emptied." if had else f"queue_clear: list {name!r} was already empty."


def queue_all_stats() -> str:
    """Tab-separated ``name<TAB>length`` lines for every list (for ``list_names`` tool)."""
    with _LOCK:
        names = sorted(_STORE.keys())
        rows: List[str] = []
        for nm in names:
            rows.append(f"{nm}\t{len(_STORE[nm])}")
    if not rows:
        return "list_names: (no lists defined yet)."
    return "list_names:\n" + "\n".join(rows)


def format_show_lists() -> str:
    """Human-readable ``/queue show lists`` output."""
    with _LOCK:
        names = sorted(_STORE.keys())
        if not names:
            return "(no lists)"
        lines = [f"  {nm}  ({len(_STORE[nm])} items)" for nm in names]
    return "Lists:\n" + "\n".join(lines)


def queue_save(name: str, path: str) -> str:
    name = _norm_listname(name)
    path = scalar_to_str(path, "").strip()
    if not name:
        return "queue_save error: list name is empty."
    if not path:
        return "queue_save error: filename is empty."
    with _LOCK:
        d = _STORE.get(name)
        items = list(d) if d else []
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    except OSError as e:
        return f"queue_save error: {e}"
    return f"queue_save: wrote {len(items)} item(s) from {name!r} to {path!r}."


def queue_load(name: str, path: str) -> str:
    name = _norm_listname(name)
    path = scalar_to_str(path, "").strip()
    if not name:
        return "queue_load error: list name is empty."
    if not path:
        return "queue_load error: filename is empty."
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except OSError as e:
        return f"queue_load error: could not read file: {e}"
    except json.JSONDecodeError as e:
        return f"queue_load error: invalid JSON: {e}"
    if not isinstance(raw, list):
        return "queue_load error: JSON root must be an array of strings."
    items: List[str] = []
    for i, x in enumerate(raw):
        if isinstance(x, str):
            items.append(x)
        elif isinstance(x, (int, float, bool)):
            items.append(str(x))
        else:
            return f"queue_load error: item at index {i} is not a string."
    with _LOCK:
        _STORE[name] = deque(items)
    return f"queue_load: loaded {len(items)} item(s) into {name!r} from {path!r}."


# --- tool handlers (params dict) ---


def tool_list_add(params: dict) -> str:
    p = params if isinstance(params, dict) else {}
    name = p.get("listname") if p.get("listname") is not None else p.get("name")
    data = p.get("data") if p.get("data") is not None else p.get("value")
    return queue_add(scalar_to_str(name, ""), scalar_to_str(data, ""))


def tool_list_remove(params: dict) -> str:
    p = params if isinstance(params, dict) else {}
    name = p.get("listname") if p.get("listname") is not None else p.get("name")
    return queue_remove(scalar_to_str(name, ""))


def tool_list_peek(params: dict) -> str:
    p = params if isinstance(params, dict) else {}
    name = p.get("listname") if p.get("listname") is not None else p.get("name")
    return queue_peek(scalar_to_str(name, ""))


def tool_list_length(params: dict) -> str:
    p = params if isinstance(params, dict) else {}
    name = p.get("listname") if p.get("listname") is not None else p.get("name")
    return queue_length(scalar_to_str(name, ""))


def tool_list_clear(params: dict) -> str:
    p = params if isinstance(params, dict) else {}
    name = p.get("listname") if p.get("listname") is not None else p.get("name")
    return queue_clear(scalar_to_str(name, ""))


def tool_list_names(params: dict) -> str:
    _ = params
    return queue_all_stats()


TOOLSET = {
    "name": "queue",
    "description": "Global in-memory named FIFO lists (scratch storage between tool calls).",
    "triggers": ["queue", "scratch", "fifo", "named list"],
    "tools": [
        {
            "id": "list_add",
            "description": "Append a string to a named list (FIFO tail); same idea as /queue NAME add.",
            "aliases": ("list_put", "queue put", "enqueue"),
            "prompt_doc": (
                "list_add — parameters.listname (or name), parameters.data (or value) non-empty strings. "
                "Appends to the global in-memory list (matches REPL ``/queue <name> add``)."
            ),
            "handler": tool_list_add,
        },
        {
            "id": "list_remove",
            "description": "Remove and return the oldest item from a named list (FIFO head).",
            "aliases": ("queue pop", "dequeue"),
            "prompt_doc": (
                "list_remove — parameters.listname (or name). Returns the next string or the literal "
                f"{_EMPTY!r} if the list is missing or empty."
            ),
            "handler": tool_list_remove,
        },
        {
            "id": "list_peek",
            "description": "Return the oldest item without removing it.",
            "aliases": ("queue peek",),
            "prompt_doc": (
                "list_peek — parameters.listname (or name). Returns the next string or "
                f"{_EMPTY!r} if missing/empty."
            ),
            "handler": tool_list_peek,
        },
        {
            "id": "list_length",
            "description": "Return the number of items in a named list (0 if missing).",
            "aliases": ("queue length", "list size"),
            "prompt_doc": "list_length — parameters.listname (or name). Returns a decimal string count.",
            "handler": tool_list_length,
        },
        {
            "id": "list_clear",
            "description": "Remove all items from a named list.",
            "aliases": ("queue clear", "empty list"),
            "prompt_doc": "list_clear — parameters.listname (or name).",
            "handler": tool_list_clear,
        },
        {
            "id": "list_names",
            "description": (
                "Discovery: list every queue name and how many items it has (tab-separated lines); "
                "same information as REPL ``/queue show lists``."
            ),
            "aliases": ("list_all", "queue list", "show lists"),
            "prompt_doc": (
                "list_names — no parameters. Returns one line per list as ``name<TAB>length``, "
                "or a short message if no lists exist. Use before list_add/list_remove when list names are unknown."
            ),
            "handler": tool_list_names,
        },
    ],
}
