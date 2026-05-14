---
title: Extension settings (/set extensions)
description: Persisted JSON overrides under extensions.<id> for shipped modules such as the code pipeline; use /set save.
---

# Extension settings (`/set extensions`)

Some **loaded REPL extensions** read optional numeric or string overrides from your **prefs file** (`~/.agent.json` by default, or `--config`). They are stored under the top-level **`extensions`** object: each **`<id>`** maps to a small dict of keys (e.g. pipeline limits).

This is **not** the same as **`/load`** (which registers Python extensions for the current process). Here you are editing **saved settings** the extensions read on future runs after **`/set save`**.

---

## Commands

| Command | Purpose |
|---------|---------|
| **`/set extensions show`** | Print all extension override dicts. |
| **`/set extensions <id> show`** | Print keys for one id (e.g. `code_pipeline`). |
| **`/set extensions <id> set <key> <value>`** | Set a key (value is parsed as number, boolean, JSON literal, or plain text). |
| **`/set extensions <id> unset <key>`** | Remove a key (revert to extension defaults where applicable). |

**`<id>`** must match `^[a-zA-Z][a-zA-Z0-9_]{0,63}$` (letter first, then letters, digits, underscore; max 64 chars).

**Persist:** After changes, run **`/set save`** (see below for `full` / `--full`).

**Help:** `/set extensions help` prints a short reminder of the `extensions` JSON shape.

---

## Example: code pipeline

The shipped **`extensions/code.py`** pipeline reads **`code_pipeline`** under `extensions`. Keys include (defaults are in the extension module if unset):

- `design_review_max`, `code_test_max`, `inner_round_max`, `parse_fail_max`, `user_ask_max_len`

Example:

```text
/set extensions code_pipeline set code_test_max 8
/set save
```

Details of the pipeline: [Code pipeline extension](code-extension.md).

---

## `/set save` snapshot shape

- **`/set save`** — Writes prefs with a **minimal** delta by default (omits values that match built-in defaults where the serializer supports it).
- **`/set save full`** or **`/set save --full`** — Writes a **full** snapshot for easier diffing or hand-editing.

---

## See also

- [Settings (`/set`, `/settings`)](settings-repl.md) — all `/set` topics including `save`
- [Configuration & environment](environment.md)
- [Code pipeline extension](code-extension.md)
- [REPL extensions](repl-extensions.md) (loading `.py` extensions)
- [README — Configuration](../README.md#configuration)
