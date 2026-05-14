---
title: REPL help (/help)
description: Built-in /help listing, extension-appended lines, and per-command help topics.
---

# `/help` in the REPL

**`/help`** and **`/?`** print a **fixed list** of core slash commands, then **optional extra lines** from any loaded REPL extensions (`register_help` on the extension registry — see [REPL extensions](repl-extensions.md)).

Some lines appear **only when the host wires hooks** (for example `agent_tui.py`): `/fork`, `/fork_background`, `/kill`, `/list`, `/switch`, `/send`, and related shortcuts. In a plain stdin REPL those commands may print a “host not configured” style message instead.

There is **no** `/help <topic>` router in the session: discovery is **`/help`** plus **`<command> help`** (or `-h` / `--help` where implemented) on each feature.

---

## Per-command help (examples)

| Try | What you get |
|-----|----------------|
| **`/set help`** | Short index of `/set` topics (same dispatcher as **`/settings help`**). |
| **`/while help`** | Full `/while` grammar and judge semantics. |
| **`/compact help`** | Percent / word targets for LLM compression. |
| **`/fork help`** · **`/send help`** (TUI) | Comma-splitting and quoting rules for multi-line payloads. |
| **`/load`** (bare) | Extension load grammar, `--help`, `/load info`. |
| **`/set extensions help`** | Shape of the `extensions` prefs object. |

Tool and toolset documentation in-session: **`/set tools describe <tool-id>`** (and toolset names where supported).

---

## Related commands not duplicated on `/help`

The printed list is intentionally short. These are still valid (see [REPL session & context](repl-session.md), [Settings](settings-repl.md), README):

- **`/show`** (models, current primary/reviewer, …)
- **`/skill`** …
- **`/import FILE`** — injects file contents as the **next user turn** (different from **`/source`**, which runs lines as slash/input).

---

## See also

- [Settings (`/set`, `/settings`)](settings-repl.md)
- [Configuration & environment](environment.md)
- [`/while` loop](while-repl.md)
- [REPL extensions](repl-extensions.md)
- Source: **`agentlib/session.py`** (`/help` branch in `_execute_command_line`)
