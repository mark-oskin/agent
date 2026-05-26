---
title: Settings in the REPL (/set and /settings)
description: Topic map for persisted and session preferences; /set help is the live command list.
---

# `/set` and `/settings`

**`/settings`** is an alias for **`/set`**. Both enter the same settings dispatcher.

**Authoritative usage:** run **`/set help`** in the REPL — it tracks the code in `agentlib/session.py` (`_cmd_settings`). This page is a **compact map** so you know where to look; it does not duplicate every error message or edge case.

**Agent access:** during a chat turn the model can run most `/set` and `/show` lines via the native tool **`session_command`** (same output you would see in the REPL). Use **`/set thinking show`** for thinking status. Details: **[session-command.md](session-command.md)**.

**Config file:** defaults to **`~/.agent.json`** (override with **`--config`**). Most changes apply to the **current session** until you run **`/set save`** (see [Extension settings](extension-settings.md) for `full` / `--full` snapshot mode).

---

## Topic overview

| Topic | What it does |
|--------|----------------|
| **`/set lock`** | Permanently freeze prefs, tool allowlists, prompts, context manager knobs, and **`/mcp`** wiring **for this session** (read-only **`show`** / **`list`** / **`status`** still work). There is no unlock — start a new session to reconfigure. Forked sessions inherit the parent lock state. |
| **`/set save`** | Write prefs to disk. Optional **`full`** or **`--full`** for a complete snapshot instead of the default minimal delta. |
| **`/set model`** | Local **Ollama** model tag (`ollama.model`) when the primary backend is Ollama. For hosted APIs, switch primary first (`/set primary llm hosted …`). |
| **`/set primary llm`** | **`ollama`** (local) or **`hosted <base_url> <model> [api_key]`** — primary chat backend. Preserves `request_options` where applicable. |
| **`/set primary request_options`** | Per-primary sampling / generation map: **`show`**, **`clear`**, **`set`**, **`unset`**, **`merge`**, **`replace`** (JSON objects). |
| **`/set second_opinion llm`** | Reviewer path: **`ollama [model]`** or **`hosted <base_url> <model> [api_key]`**. |
| **`/set enable`** / **`/set disable`** | Session **features** only: **`second_opinion`**, **`stream_thinking`**, **`stream_assistant`**, **`show_draft`**, **`verbose`**. Tool ids still work here but prefer **`/set tools …`**. |
| **`/set tools`** | **`list`**, **`<tool or toolset> enable|disable`**, **`reload`**, **`describe` `<tool-id>`** (or toolset name). |
| **`/mcp`** | Model Context Protocol — **`help`**, **`list`**, **`status`**, **`add`** (optional **`--framing ndjson`** for Python SDK stdio), **`remove`**, **`enable`** / **`disable`**, **`reload`**. See [Configuration & environment](environment.md#mcp-model-context-protocol). |
| **`/set system_prompt`** | **`show`**, **`reset`**, **`pin`** (snapshot effective prompt), **`file`**, **`save`**, or inline text. |
| **`/set prompt_template`** | **`list`**, **`show`**, **`use`**, **`default`**, **`set`**, **`delete`** — JSON templates under `prompt_templates_dir` / overlays. |
| **`/set context`** | Context-window manager: **`show`**, **`on`/`off`**, **`tokens`**, **`trigger`**, **`target`**, **`keep_tail`**. Also see **`agent.disable_context_manager`** in prefs ([Configuration & environment](environment.md)). |
| **`/set thinking`** | Ollama **`think`** field: **`show`**, **`on`/`off`**, **`level low|medium|high`** (also toggles **`stream_thinking`** as documented in `/set thinking` help). |
| **`/set verbose`** | **`0`–`3`** or **`on`/`off`** — logging / banner verbosity. |
| **`/set extensions`** | Extension-specific prefs under `extensions.<id>` — see [Extension settings](extension-settings.md). |
| **`/set ollama`**, **`/set openai`**, **`/set agent`** | **Group** prefs: **`show`**, **`keys`**, **`set <name> <value>`**, **`unset <name>`** (lowercase keys; values parsed as numbers, booleans, JSON, or text). |

---

## Tips

- After changing groups (`ollama`, `openai`, `agent`), tools, templates, or context — run **`/set save`** if you want the JSON file updated.
- **`/set tools describe <id>`** prints tool contract text when the registry has it (core + plugins).
- Unknown **`/set …`** lines end with “Try **`/help`**” from the dispatcher.

---

## See also

- [REPL help (`/help`)](help-repl.md)
- [Configuration & environment](environment.md)
- [Extension settings (`/set extensions`)](extension-settings.md)
- [README — Configuration](../README.md#configuration)
- [session_command](session-command.md) · [Core tools](core-tools.md) · [Plugin toolsets](plugin-toolsets.md)
- Implementation: **`agentlib/session.py`** (`_cmd_settings`, `_cmd_set_extensions`, …)
