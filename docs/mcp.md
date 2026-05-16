---
title: MCP (Model Context Protocol)
description: Plug in external tools from MCP servers over stdio or HTTP; REPL /mcp commands, framing, and prefs.
---

# MCP (Model Context Protocol)

The agent can merge **external tools** from [Model Context Protocol](https://modelcontextprotocol.io/) servers into the same tool loop as built-ins and plugins. Each MCP tool is exposed to the model as **`mcp_<server>_<tool>`** (see **`/set tools describe`** when that id is in the session allowlist).

## Process-wide vs per session

| Layer | What it controls |
|-------|------------------|
| **`agent.mcp_enabled`** + **`agent.mcp_servers`** | One shared MCP connection per process (**`AgentApp`**): subprocesses / HTTP, tool discovery into **`mcp_registry`**. |
| **This session's `enabled_tools`** | Which discovered **`mcp_*`** tools the model may call in **this** REPL lane or embed session. |

**`/mcp enable`** connects servers for the whole process. **`/mcp session on`** enables all currently discovered MCP tool ids **in this session only**. **`/fork`** copies the parent's **`enabled_tools`** (including MCP ids). Other lanes are unaffected unless they run **`/mcp session on`** themselves.

Use **`/set tools`** to see MCP tools with **`[on]`** / **`[off]`** per session, or **`/set enable`** / **`disable`** on individual **`mcp_*`** ids.

## REPL commands

Use **`/mcp help`** for syntax. In short:

- **`/mcp enable`** / **`disable`** — start or stop **shared** MCP servers (**`agent.mcp_enabled`**).
- **`/mcp session on`** / **`off`** — add or remove all discovered **`mcp_*`** tools from **this session's** allowlist.
- **`/mcp add stdio …`** / **`add http …`** — append or upsert a server entry in **`agent.mcp_servers`**.
- **`/mcp list`** / **`status`** — prefs, errors, discovered count, and how many MCP tools are on in this session.
- **`/mcp remove NAME`**, **`reload`** — drop a server or schedule a reconnect.

Persist prefs with **`/set save`** (writes **`~/.agent.json`** or your **`--config`** file). Per-session **`enabled_tools`** are in-memory unless you save a context bundle that includes them.

## Transports

| Transport | Role |
|-----------|------|
| **stdio** | Local subprocess; JSON-RPC over stdin/stdout. |
| **http** | JSON-RPC POST to a fixed URL (simple request/response). |

## Stdio framing (important)

Two wire formats exist for stdio MCP:

| Mode | When to use |
|------|-------------|
| **`content-length`** (default) | **`Content-Length:`** header + JSON body — typical for **Node / `npx`** servers. |
| **`ndjson`** | One JSON object per line — required for many **Python MCP SDK** / **FastMCP** servers. |

In the REPL, pass **`--framing ndjson`** on **`/mcp add stdio …`**. In JSON prefs, set **`stdio_framing`** (alias **`framing`**) on that server object. If the server logs parse errors on **`Content-Length:`** lines, switch to **`ndjson`**.

## Python servers

Use the **interpreter that has the `mcp` package** (usually your project **`.venv/bin/python3`**) in **`command`**, not bare **`python`** on PATH. The agent sets **`PYTHONUNBUFFERED`** and inserts **`-u`** for Python launchers to reduce stdio buffering issues.

## Resync and discovery

With MCP enabled, reconnects after **`/mcp`** or relevant **`/set agent …`** changes are **scheduled in the background** so the REPL returns immediately. Use **`/mcp status`** or watch **`stderr`** for **`[mcp]`** lines to confirm tools and errors.

---

For prefs shape, **`env`**, HTTP headers, and environment-variable notes, see **[Configuration & environment](environment.md)** (MCP section).
