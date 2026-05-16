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

## MCP (Model Context Protocol)

Configure optional MCP servers **from the REPL** with **`/mcp`** (`help`, `list`, `status`, `add`, `remove`, `enable` / `disable`, `reload`). That updates in-memory prefs (**`agent.mcp_enabled`**, **`agent.mcp_servers`**); use **`/set save`** to persist **`~/.agent.json`**.

**`agent.mcp_enabled`** must be **`true`** (via **`/mcp enable`** or **`/set agent set mcp_enabled true`**) before subprocesses start and tools are discovered. Adding a server only updates prefs; it does not enable MCP by itself.

**Per session:** even with servers connected, the model only receives MCP tools that are in **this session's** **`enabled_tools`** (use **`/mcp session on`**, **`/mcp session off`**, or **`/set tools`**). **`/fork`** copies the parent's tool allowlist. See **[MCP (Model Context Protocol)](mcp.md)**.

You can still edit JSON by hand or use **`/set agent set mcp_servers '…'`** if you prefer.

### REPL resync behavior

After **`/mcp`**, **`/set agent …`** (when keys include **`mcp_enabled`** or **`mcp_servers`**), or **`/set tools reload`**, MCP reconnect is **scheduled in the background** when MCP is enabled so the **`>`** prompt returns immediately. Watch **`stderr`** for **`[mcp] Background reconnect complete…`** lines, or run **`/mcp status`** for tool counts and last connection errors.

Server entry shape (what **`/mcp add`** writes, plus optional fields you can add in JSON):

| Field | Meaning |
|-------|---------|
| **`name`** | Short id (`a-z`, digits, underscore). |
| **`transport`** | **`stdio`** or **`http`**. |
| **stdio:** **`command`**, **`args`**, optional **`cwd`**, **`env`**, optional **`stdio_framing`** (or alias **`framing`**). |
| **http:** **`url`**, optional **`headers`**. |

### stdio JSON-RPC framing

Two wire formats are supported for **stdio** transports:

| **`stdio_framing` value** | Use when |
|---------------------------|----------|
| **`content-length`** (default) | **`Content-Length:`** header + JSON body (common for **Node / `npx`** MCP servers and many TypeScript clients). |
| **`ndjson`** | One UTF-8 **JSON object per line** (newline-terminated). Required for servers built on the **Python MCP SDK** / **FastMCP** stdio transport. |

In the REPL, pass **`--framing ndjson`** on **`/mcp add stdio …`** to store **`stdio_framing`** in that server entry. If framing is wrong, the server may log JSON parse errors on **`Content-Length:`** lines and the client may time out on **`initialize`** until you fix it.

### Python subprocess tips

- Use the **project venv interpreter** in **`command`** (e.g. **`./.venv/bin/python3 -m mypackage.server`**) so imports such as **`mcp`** resolve; bare **`python`** on **`PATH`** is often system Python without those packages.
- The agent sets **`PYTHONUNBUFFERED=1`** for stdio children unless **`env`** in the server entry explicitly sets **`PYTHONUNBUFFERED`**, and inserts **`-u`** after **`python` / `python3` / `python3.x`** launchers when **`-u`** is not already present.

Examples:

```text
/mcp add stdio fs --cwd /tmp npx -y @modelcontextprotocol/server-filesystem /tmp
/mcp add stdio pyapi --cwd /path/to/proj --framing ndjson ./.venv/bin/python3 -m mymcp.server
/mcp add stdio desk --cwd /path/to/proj --framing ndjson /path/to/proj/.venv/bin/my-mcp-entrypoint
/mcp add http api https://example.com/mcp --header "Authorization=Bearer TOKEN"
/mcp enable
/mcp status
```

After **`mcp_enabled`** or **`mcp_servers`** changes via **`/set agent …`**, MCP resync runs as above; **`/set tools reload`** also schedules MCP refresh together with plugin toolsets.

**HTTP** mode expects **JSON-RPC POST** request/response; streaming-only gateways may not work.

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
