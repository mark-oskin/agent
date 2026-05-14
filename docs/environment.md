---
title: Configuration and environment
description: Prefs JSON vs process environment; known AGENT_* variables read by this repo.
---

# Configuration and environment

Most behavior is controlled by **`~/.agent.json`** (or **`--config path.json`**), loaded into **`AgentSettings`** (`ollama`, `openai`, `agent`, `extensions` groups). See [Settings (`/set`)](settings-repl.md) for changing values in the REPL.

Legacy prefs may store **env-style key names** inside those JSON groups (for example `OLLAMA_HOST` as an alias for `host`); that is **prefs migration**, not automatic reading of the shell environment.

---

## Environment variables used by core code

The main CLI does **not** read **`AGENT_BROWSER_ENGINE`** or **`BRAVE_SEARCH_API_KEY`** from the process environment. Configure the Playwright default engine with prefs **`agent.default_browser_engine`** (`chromium` / `webkit` / `safari` aliases as in `tools/web.py`), and the Brave Search API key with **`agent.brave_search_api_key`** when **`agent.search_web_backend`** is **`brave`**.

No other **`AGENT_*`** names are read via `os.environ` in **`agentlib/`** for the main CLI today. Directory overrides for tools and skills are the prefs keys **`agent.tools_dir`** and **`agent.skills_dir`** (or top-level snapshot fields accepted on load — see `agentlib/prefs/bootstrap.py`), not separate `AGENT_TOOLS_DIR` / `AGENT_SKILLS_DIR` env vars.

---

## Context manager vs “AGENT_CONTEXT_*”

Automatic transcript compaction honors:

- The **`context_manager`** object in prefs (`enabled`, `tokens`, `trigger_frac`, `target_frac`, `keep_tail_messages`, …), adjustable with **`/set context …`**.
- The boolean **`agent.disable_context_manager`** in settings (`DEFAULT_SETTINGS` in `agentlib/settings.py`); when true, compaction is skipped (`agentlib/context/compaction.py`).

There is **no** `/help environment` command. Inspect current values with **`/set context show`** and **`/set agent show`** (look for `disable_context_manager` and related keys).

---

## CLI vs REPL

- **`--config`**, **`-enable-tool` / `-disable-tool`**, **`--model`**, **`--cloud-ai`**, etc. apply at process start; see `./agent.py --help`.
- REPL **`/set save`** writes the prefs file for future runs (minimal vs full snapshot — [Extension settings](extension-settings.md)).

---

## See also

- [REPL help (`/help`)](help-repl.md)
- [README — Configuration](../README.md#configuration)
- [README — CLI options](../README.md#cli-options-summary)
