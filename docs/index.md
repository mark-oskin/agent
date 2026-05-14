---
title: Documentation index
description: User-facing guides for core tools, plugin toolsets, REPL control flow, /set topics, help, environment, extension prefs, and bundled extensions.
---

# Documentation

These pages complement [README.md](../README.md) with deeper, website-style guides. The REPL’s `/help` lists slash commands; **`/set help`** and per-command **`… help`** cover settings (see [REPL help](help-repl.md)).

| Topic | Description |
|--------|-------------|
| [Core tools](core-tools.md) | Every built-in tool id, role, and parameter summary |
| [Plugin toolsets](plugin-toolsets.md) | Bundled optional toolsets (`dev`, `desktop`, `browser`, …) and how to enable them |
| [REPL extensions](repl-extensions.md) | `/load`, `/unload`, `/extensions`, `/tokens`, extension API sketch |
| [REPL session & context](repl-session.md) | `/cd`, `/context`, `/load_context`, `/save_context`, `/source`, JSON bundle format |
| [Queue toolset and `/queue`](queue.md) | Named FIFO lists for the model and optional REPL commands |
| [Code pipeline extension](code-extension.md) | `/load extensions/code.py`, `/code`, multi-lane vs single-lane |
| [`/while` loop](while-repl.md) | Model-judged condition loop, `--max`, comma-separated body prompts |
| [Extension settings (`/set extensions`)](extension-settings.md) | Prefs keys under `extensions.<id>` (e.g. `code_pipeline`), `/set save` minimal vs full |
| [Settings (`/set`, `/settings`)](settings-repl.md) | Topic map: tools, primary LLM, context manager, templates, `save` — use `/set help` for full syntax |
| [REPL help (`/help`)](help-repl.md) | Built-in listing, extension lines, `<command> help` pattern |
| [Configuration & environment](environment.md) | Prefs JSON vs env; `AGENT_BROWSER_ENGINE`; context manager keys |
