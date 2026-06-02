---
title: REPL session — cwd, context files, /source
description: Working directory, saving and loading chat JSON, auto-save logs, and batching REPL lines from a file.
---

# REPL session: cwd, context, `/source`

These commands affect **where tools run** and how the **transcript** is stored on disk. They complement `/clear`, `/compact`, and `/settings` (see `/help` for the full command list).

---

## Working directory: `/cd` · `/chdir`

- **Usage:** `/cd <dir>` (path tokens after the command are joined; relative segments resolve against the **session** cwd, not the process cwd).
- **Effect:** Updates `session_cwd` for **`run_command`**, **`!` shell escapes**, and filesystem-related tools that honor the session workspace.
- On success, the REPL prints the new working directory.

---

## Context bundles: `/context` · `/load_context` · `/save_context`

### JSON format

Load/save logic lives in **`agentlib/context/io.py`**. A file may be:

1. A **JSON array** of message objects: `[{"role":"user","content":"…"}, …]`, or  
2. A **bundle object**: `{"messages": [ … ], …}`.

Each message must be a dict with **`role`** one of `user`, `assistant`, `system`, or **`tool`** (native tool results), and **`content`** (string or coerced to string). Assistant rows may include **`tool_calls`** (Ollama/native shape); tool rows may include **`tool_name`** and **`tool_call_id`**. Non-dict entries are skipped; invalid roles are rejected (see `parse_context_messages_data` in `agentlib/context/io.py`).

**Written bundles** (snapshots from this agent) also include metadata:

| Field | Meaning |
|--------|---------|
| `version` | Schema version (currently `1`) |
| `user_query` | Last user line when saved (may be empty for manual snapshots) |
| `final_answer` | Last assistant answer when known |
| `answered` | Whether the last turn completed with an answer |
| `messages` | Full chat list |

### Commands

| Command | Behavior |
|---------|-----------|
| **`/context load FILE`** | Replace in-memory `messages` with the file contents. Alias: **`/load_context FILE`** (note: `/load_context` takes the **rest of the line** as the path—useful for paths with spaces). |
| **`/context save FILE`** | Write **one** snapshot of the current transcript; does **not** enable auto-save. Alias: **`/save_context FILE`** (same “rest of line” path rule). |
| **`/context start_log FILE`** | Write the current transcript, then set **`session_save_path`** so that after each **normal** conversational turn the bundle is written again (same `save_context_bundle` helper with updated `user_query` / `final_answer` / `answered`). |

Paths are passed through `os.path.expanduser`.

**CLI (one-shot):** `--load-context <file>` loads a bundle before your question; `--save-context <file>` writes after the run. `--load-context` requires a non-empty question on the same invocation (see `agentlib/app.py` help text).

---

## `/source`

- **Usage:** `/source <file>`  
- Reads the file as **UTF-8**, skips blank lines, and runs each remaining line through **`execute_line`** as if you had typed it (similar to shell `source`).
- **Printed output** from each line (slash-command text and **`SessionLineResult.output`**) is shown as it runs — same as typing the lines interactively.
- If a sourced line triggers **`/quit`**, sourcing stops and the session exits.
- On completion, prints how many lines were executed.

Use for reproducible setup scripts (e.g. a sequence of `/settings` and `/load` lines).

---

## `/last` (multi-agent hosts)

**`/last answer [NAME]`** and **`/last question [NAME]`** forward to the host (`agent_tui`, etc.) when wired. In a plain stdin REPL they may no-op or print a host hint depending on configuration.

---

## See also

- [REPL extensions](repl-extensions.md) (`/load` Python extensions — different from **context** JSON)
- [`/while` loop](while-repl.md)
- [Extension settings (`/set extensions`)](extension-settings.md)
- [Settings (`/set`, `/settings`)](settings-repl.md)
- [REPL help (`/help`)](help-repl.md)
- [Configuration & environment](environment.md)
- [README — Interactive REPL](../README.md#interactive-repl-important-commands)
- Source: **`agentlib/session.py`** (`_cmd_cd`, `_cmd_context`, `_cmd_load_context`, `_cmd_save_context`, `_cmd_source`)
