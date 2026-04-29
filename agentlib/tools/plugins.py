from __future__ import annotations

import importlib
import importlib.util
import os
from typing import Callable, Optional


# Plugin toolsets (loaded from tools/ directory).
# Toolsets are off by default and can be enabled by the user.
PLUGIN_TOOLSETS: dict[str, dict] = {}
PLUGIN_TOOL_HANDLERS: dict[str, Callable[[dict], str]] = {}
PLUGIN_TOOL_TO_TOOLSET: dict[str, str] = {}
PLUGIN_TOOLSET_TRIGGERS: dict[str, list[str]] = {}


def load_plugin_toolsets(*, tools_dir: Optional[str], default_tools_dir: str) -> None:
    """
    Load plugin toolsets from a tools directory.

    Each plugin module must define:
      TOOLSET = {
        "name": "dev" | "web" | ...,
        "description": "…",
        "triggers": ["keyword", "regex:..."] (optional),
        "tools": [
          {"id": "run_pytest", "description": "...", "aliases": ["pytest", ...], "handler": callable},
          ...
        ],
      }
    """
    PLUGIN_TOOLSETS.clear()
    PLUGIN_TOOL_HANDLERS.clear()
    PLUGIN_TOOL_TO_TOOLSET.clear()
    PLUGIN_TOOLSET_TRIGGERS.clear()
    base0 = (tools_dir or "").strip()
    base = os.path.abspath(os.path.expanduser(base0)) if base0 else default_tools_dir
    if not os.path.isdir(base):
        return
    use_pkg_import = os.path.abspath(base) == os.path.abspath(default_tools_dir)
    for fn in sorted(os.listdir(base)):
        if not fn.endswith(".py") or fn.startswith("_"):
            continue
        modname = os.path.splitext(fn)[0]
        try:
            if use_pkg_import:
                m = importlib.import_module(f"tools.{modname}")
            else:
                path = os.path.join(base, fn)
                spec = importlib.util.spec_from_file_location(f"agent_tools_{modname}", path)
                if spec is None or spec.loader is None:
                    continue
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
        except Exception:
            continue
        ts = getattr(m, "TOOLSET", None)
        if not isinstance(ts, dict):
            continue
        nm = str(ts.get("name") or "").strip().lower()
        if not nm:
            continue
        tools = ts.get("tools")
        if not isinstance(tools, list) or not tools:
            continue
        PLUGIN_TOOLSETS[nm] = ts
        tr = ts.get("triggers")
        if isinstance(tr, list):
            PLUGIN_TOOLSET_TRIGGERS[nm] = [str(x) for x in tr if str(x).strip()]
        else:
            PLUGIN_TOOLSET_TRIGGERS[nm] = []
        for td in tools:
            if not isinstance(td, dict):
                continue
            tid = str(td.get("id") or "").strip()
            if not tid:
                continue
            h = td.get("handler")
            if not callable(h):
                continue
            PLUGIN_TOOL_HANDLERS[tid] = h
            PLUGIN_TOOL_TO_TOOLSET[tid] = nm


def plugin_tools_for_toolset(toolset: str) -> set[str]:
    nm = (toolset or "").strip().lower()
    out: set[str] = set()
    for tid, ts in PLUGIN_TOOL_TO_TOOLSET.items():
        if ts == nm:
            out.add(tid)
    return out


def plugin_tool_entries() -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    entries = []
    for ts in PLUGIN_TOOLSETS.values():
        tools = ts.get("tools") if isinstance(ts, dict) else None
        if not isinstance(tools, list):
            continue
        for td in tools:
            if not isinstance(td, dict):
                continue
            tid = str(td.get("id") or "").strip()
            if not tid:
                continue
            desc = str(td.get("description") or "").strip() or "Plugin tool"
            aliases = td.get("aliases")
            if not isinstance(aliases, (list, tuple)):
                aliases = ()
            aliases_t = tuple(str(a) for a in aliases if str(a).strip())
            entries.append((tid, desc, aliases_t))
    return tuple(entries)

