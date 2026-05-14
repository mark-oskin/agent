---
title: REPL extensions (/load)
description: Load optional Python REPL extensions, inspect them, and unload; same mechanism powers /queue and the code pipeline.
---

# REPL extensions (`/load`)

REPL **extensions** are ordinary **Python files** you load at runtime. They register slash commands and optional `/help` lines without restarting the agent.

---

## Commands

| Command | Purpose |
|---------|---------|
| `/load FILE.py [ … --flags ]` | Execute the module and call `register_repl(session, registry)`. Path is resolved relative to the session working directory unless absolute. |
| `/load FILE.py --help` or `/load FILE.py -h` | Print load options for that file **without** registering (via `describe_repl_load_options()` if defined). |
| `/load info FILE.py` | Same documentation path as `--help`. |
| `/unload` | Remove **all** loaded extensions and their slash commands. |
| `/extensions` | List loaded extension paths, registered command names, and normalized load flags. |
| `/tokens` | Alias for **`/usage`**: show last Ollama prompt/completion token usage when available. |

Bare **`/load`** prints built-in usage: `register_repl`, `registry.register_command`, `registry.register_help`, post-load return values, and the `--help` / `info` forms.

---

## Load flags

Tokens after the **first `--` that follows the file path** are **load options**. They are normalized to names on `registry.load_flags` (for example `--single-lane` → `single_lane`). Extensions read these in `register_repl` to change behavior. Document per-extension flags with `describe_repl_load_options() -> str` in the same file so `/load path --help` stays accurate.

---

## Author contract (sketch)

- **`register_repl(session, registry)`** (required): register handlers with `registry.register_command("name", handler)` for `/name …` lines and optional `registry.register_help("…")` lines appended to `/help`.
- **`describe_repl_load_options() -> str`** (optional): human text for `/load … --help` and `/load info …`.
- **Return value:** `None`, a single `str`, or a `list[str]` of REPL lines to run via `execute_line` after a successful load.

Bundled examples: `extensions/queue_control.py`, `extensions/code.py` (see [Code pipeline extension](code-extension.md)).

---

## See also

- [REPL session & context](repl-session.md) (`/cd`, transcript save/load — not Python extensions)
- [Extension settings (`/set extensions`)](extension-settings.md) (persisted prefs for extensions such as `code_pipeline`)
- [Settings (`/set`, `/settings`)](settings-repl.md) (tools, LLM routing, context manager, `save`, …)
- [Plugin toolsets](plugin-toolsets.md) (LLM tools from `tools/`, not the same as `/load` extensions)
- [Queue toolset and `/queue`](queue.md)
- [Code pipeline extension](code-extension.md)
