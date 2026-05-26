---
title: Core tools
description: Built-in tool ids always available to the model; disable per session via settings or CLI flags.
---

# Core tools

**Core** tools are defined in code (not loaded from `tools/`). They can be **disabled** for a session; blocked calls return an error string to the model.

**Discover:** `./agent.py --list-tools` (or `agent --list-tools`) prints ids and on/off state. In the REPL, `/settings tools` also lists **plugin** toolsets.

**Aliases:** Natural phrases map to ids (for example “web search” → `search_web`). The authoritative alias table lives in `agentlib/tools/routing.py` as `CORE_TOOL_ENTRIES`.

---

## Tool table

| ID | Role |
|----|------|
| `search_web` | Web search (DuckDuckGo HTML); optional `max_results` (1–30). |
| `search_web_fetch_top` | Same search plus **fetched excerpts** from top result pages; optional `max_results`, `fetch_top_n` (1–10). When both this and `search_web` are enabled, routing prefers `search_web` unless the user intent matches this tool’s triggers. |
| `fetch_page` | HTTP GET one or more URLs (`url` and/or `urls`); combined text with batch limits. |
| `run_command` | Run a shell command (`command`). |
| `use_git` | Vetted git operations (`op`, paths, message, remote, branch, …). |
| `write_file` | Write or overwrite a file (`path`, `content`). |
| `read_file` | Read a file (`path`). |
| `grep` | Regex search over a file or directory (`pattern`, optional `path`, `glob_pattern`, `max_matches`, `max_files`, `ignore_case`). |
| `list_directory` | List a directory (`path`). |
| `download_file` | Download a URL to a path (`url`, `path`). |
| `tail_file` | Read the end of a file (`path`, optional `lines`). |
| `replace_text` | Regex replace in a file (`path`, `pattern`, `replacement`, optional `replace_all`). |
| `call_python` | Run Python in-process (`code`, optional `globals`); output includes prints and a JSON summary of assigned locals. Does **not** define `ai` / `session` — use `session_command` for slash lines. |
| `session_command` | Run one REPL slash command (`command`); returns captured command output. See **[session-command.md](session-command.md)**. |

---

## Parameter details

The strings the model sees under “Parameters per tool” are maintained in `CORE_TOOL_PROMPT_DOCS` in `agentlib/tools/routing.py`. If behavior differs from a summary here, treat that dict as source of truth.

**See also:** [session_command](session-command.md), [Plugin toolsets](plugin-toolsets.md), [REPL session & context](repl-session.md), [Settings (`/set`)](settings-repl.md), [README § Tools](../README.md#tools).
