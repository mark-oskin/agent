---
title: Queue toolset and /queue extension
description: Global in-memory named FIFO lists—LLM tools in the queue toolset and optional /queue REPL commands sharing one store.
---

# Queue toolset and `/queue` extension

The **queue** feature gives you **named FIFO lists** held in memory inside the agent process. Items are plain **strings** (you can store JSON as a string if you need structure).

Two surfaces use the **same storage**:

| Surface | Purpose |
|--------|---------|
| **Queue toolset** (`tools/queue.py`) | LLM **tool calls** (`list_add`, `list_remove`, …) when the toolset is enabled |
| **Queue extension** (`extensions/queue_control.py`) | Human-oriented **`/queue`** slash commands after you `/load` the extension |

There is **no disk persistence** unless you explicitly **`save`** / **`load`** a list to a JSON file (REPL only for those two operations).

---

## Enabling the LLM toolset

The toolset id is **`queue`**. It is **off** until you enable it (same rules as other plugin toolsets).

**REPL (typical):**

```text
/set tools queue enable
```

**Notes:**

- Toolsets live under the project’s `tools/` directory. This repo ships **`tools/queue.py`**.
- After adding or editing a plugin file, run **`/settings tools reload`** so imports refresh.
- If **several** toolsets are enabled, the agent may only attach toolsets whose **triggers** match the current user message (see main docs on plugin routing). If **only** `queue` is enabled, it stays active for every request.
- Trigger phrases registered for `queue` include: `queue`, `scratch`, `fifo`, `named list`.

Use **`/settings tools describe list_add`** (or another tool id / `queue`) for the built-in parameter summary.

---

## LLM tools (toolset `queue`)

All tools take a JSON **`parameters`** object. List names and payloads are coerced to strings (models sometimes emit numbers or small structures).

### Common parameters

| Parameter | Used by | Meaning |
|-----------|---------|---------|
| **`listname`** or **`name`** | All except `list_names` | Logical list id (e.g. `todos`, `urls`). |
| **`data`** or **`value`** | `list_add` | Non-empty string to append (FIFO **tail**). |

### Tool reference

| Tool id | What it does |
|---------|----------------|
| **`list_add`** | Append one string to the named list (same as **`/queue NAME add`**). Returns a short confirmation including the new length. |
| **`list_remove`** | **Pop** the oldest item (**FIFO head**). Returns that string, or the literal `` `<empty>` `` if the list is missing or empty. |
| **`list_peek`** | Read the oldest item **without** removing it. Returns the string or `` `<empty>` ``. |
| **`list_length`** | Returns the item count as a decimal string (`0` if the list does not exist). |
| **`list_clear`** | Drops all items for that name (list may remain as empty). |
| **`list_names`** | **Discovery:** no parameters. Returns one line per list: list name, a **tab** character, then the length (or a short line if there are no lists). Use when the model does not yet know which names exist—same information as **`/queue show lists`**, different formatting. |

Phrase aliases (for settings / routing phrases, not a substitute for the canonical JSON tool id) include e.g. **`list_put`** → `list_add`, **`list_all`** → `list_names`, plus `enqueue`, `dequeue`, `queue peek`, etc.—see `tools/queue.py` for the full `TOOLSET`.

### Empty sentinel

For **`list_remove`** and **`list_peek`**, an empty or missing queue is reported exactly as:

```text
<empty>
```

Document this in prompts so the model does not treat it as an error string unless you define otherwise.

---

## REPL extension (`/queue`)

Load once per session (path is from your working directory or absolute):

```text
/load extensions/queue_control.py
```

### Help line

```text
/queue show lists — /queue NAME add | remove | peek | length | clear | save | load
```

Use **`/queue help`** (or `-h` / `--help`) to print that line again.

### Commands

| Command | Behaviour |
|---------|-----------|
| **`/queue show lists`** | Prints each list name and item count (human-readable). Same underlying data as tool **`list_names`**. |
| **`/queue NAME add …`** | Appends the remainder of the line (after `add`) as one string—use **shell-style quoting** if you need spaces or special characters (`shlex`). |
| **`/queue NAME remove`** | Same as **`list_remove`**: pops the FIFO head, or prints `` `<empty>` ``. |
| **`/queue NAME peek`** | Same as **`list_peek`**. |
| **`/queue NAME length`** | Prints `NAME: N item(s)` using **`list_length`**. |
| **`/queue NAME clear`** | Same as **`list_clear`**. |
| **`/queue NAME save FILE`** | Writes the list to **JSON**: a single array of strings, in FIFO order (head first). Exactly **one** path token after `save`; quote the path if it contains spaces. |
| **`/queue NAME load FILE`** | Replaces that list with the contents of a JSON file (must be a JSON **array**; each element becomes a string). Same quoting rules as `save`. |

**Paths for `save` / `load`** are resolved with the session **working directory** (change with **`/cd`** in the REPL), like other workspace-relative tools.

---

## Semantics (FIFO)

- **`list_add`** / **`/queue … add`** always push on the **tail**.
- **`list_remove`** / **`/queue … remove`** always take from the **head** (oldest item).

Order is stable until you add more items or clear/replace the list.

---

## Scope and limitations

- **Process-global:** every session and lane that shares the same Python process sees the **same** lists. Names are a shared namespace—pick distinctive list names in multi-session setups.
- **In-memory:** data is lost when the agent process exits, unless you **`save`** to disk.
- **No built-in quotas:** long lists or huge strings consume RAM like any other in-memory structure.

---

## Unloading

- **Extension:** `/unload` removes REPL slash handlers from loaded extension modules (see [REPL extensions](repl-extensions.md)).
- **Toolset:** disable with **`/set tools queue disable`** (or adjust enabled toolsets in your config).

---

## See also

- **[Documentation index](index.md)** — other guides (`/load`, core tools, plugin toolsets, code pipeline).
- **[REPL extensions](repl-extensions.md)** — `/load`, `/unload`, `/extensions`.
- **[REPL session & context](repl-session.md)** — `/cd`, `/context`, `/source`.
- Repository **`README.md`** — plugin toolsets, `/settings tools`, routing.
- Source: **`tools/queue.py`**, **`extensions/queue_control.py`**.
