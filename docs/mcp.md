---
title: MCP (Model Context Protocol)
description: Plug in external tools from MCP servers over stdio or HTTP; REPL /mcp commands, framing, and prefs.
---

# MCP (Model Context Protocol)

The agent can merge **external tools** from [Model Context Protocol](https://modelcontextprotocol.io/) servers into the same tool loop as built-ins and plugins. Each MCP tool is exposed to the model as **`mcp_<server>_<tool>`** (see **`/set tools describe`** when MCP is active).

## REPL commands

Use **`/mcp help`** for syntax. In short:

- **`/mcp enable`** / **`disable`** — turn MCP on or off (**`agent.mcp_enabled`**). Tools are only started when MCP is enabled.
- **`/mcp add stdio …`** / **`add http …`** — append or upsert a server entry in **`agent.mcp_servers`**.
- **`/mcp list`** / **`status`** — show prefs, last errors, and how many tools were discovered in this process.
- **`/mcp remove NAME`**, **`reload`** — drop a server or schedule a reconnect.

Persist changes with **`/set save`** (writes **`~/.agent.json`** or your **`--config`** file).

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
