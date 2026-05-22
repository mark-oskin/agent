---
title: Plugin toolsets
description: Optional LLM tools from the tools/ directory; enable with /settings tools, routing by triggers, reload after new files.
---

# Plugin toolsets

**Plugin toolsets** add tools from Python modules under a **tools directory**. They are **off** until you enable them.

**Default directory:** the repo’s `tools/` package. **Override:** `agent.tools_dir` in your config JSON (`/set agent set tools_dir …` then `/set save`; see [Configuration & environment](environment.md)).

Each module exports a **`TOOLSET`** dict: `name`, `description`, optional `triggers`, and a list of tools (`id`, `description`, `aliases`, `handler`, optional `params` / `prompt_doc`).

---

## Enabling, routing, reload

- **Enable:** `/set tools <toolset-name> enable` (e.g. `/set tools queue enable`).
- **Routing:** With **multiple** toolsets enabled, the agent tries to expose only toolsets whose **triggers** match the current user message (smaller tool list). With **one** toolset enabled, it is always active. If nothing matches, all enabled toolsets are considered (fallback).
- **Reload:** After adding a new `.py` file, `/settings tools reload` refreshes imports without restarting.
- **Details:** `/settings tools describe <tool-id>` or `<toolset-name>` for parameters where documented.

**CLI:** `-enable-tool` / `-disable-tool` accept tool ids.

---

## Bundled toolsets (this repo)

| Toolset | Tools / notes |
|---------|----------------|
| `dev` | `run_pytest` — test suite (`uv run pytest` when `uv.lock` exists, else `python3 -m pytest`). |
| `desktop` | `open_url` — macOS `open` in the default browser (`http`/`https` only). |
| `queue` | `list_add`, `list_remove`, `list_peek`, `list_length`, `list_clear`, `list_names` — see [Queue toolset and `/queue`](queue.md). |
| `browser` | Playwright-backed automation: `browser_navigate`, `browser_click`, `browser_fill`, `browser_type`, `browser_press`, `browser_snapshot`, `browser_wait`, `browser_close`. Engines include **chromium** (default) and **webkit** (Safari engine, not Safari.app). Default when the model omits `engine` / `browser`: prefs **`agent.default_browser_engine`**. Install with **`uv sync --extra browser`**; without Playwright, enable fails with a clear message. |
| `applescript` | `run_applescript` — runs AppleScript via `osascript` (timeouts, optional script echo, temp file mode for better error locations). **macOS only; side effects.** |
| `lanes` | `agent_send` — in **agent_tui**, send one REPL line to another lane (optional `wait` / `timeout_ms`). In a plain stdin REPL, behavior is limited to what the host wires. |

---

## REPL extensions vs plugin toolsets

| Mechanism | What it adds | How you turn it on |
|-----------|----------------|---------------------|
| **Plugin toolset** | **LLM** `tool_call` handlers | `/set tools <name> enable` |
| **`/load` extension** | **Slash** commands (`/queue`, `/code`, …) | `/load path/to/extension.py` |

Some features use **both** (for example **queue**: toolset `queue` + optional `/load extensions/queue_control.py`).

---

## See also

- [Core tools](core-tools.md)
- [REPL extensions](repl-extensions.md)
- [REPL session & context](repl-session.md) (`/cd`, workspace-relative paths)
- [Settings (`/set`, `/settings`)](settings-repl.md)
- [Configuration & environment](environment.md)
- [README § Tools](../README.md#tools)
