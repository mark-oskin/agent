---
title: session_command (agent runs REPL slash commands)
description: Native tool that executes REPL slash lines and returns captured output to the model.
---

# `session_command` — agent REPL control

During a chat turn, the model can run the same **slash commands** you would type in the REPL by calling the native tool **`session_command`** with a **`command`** string (one line, starting with `/`).

Implementation:

- **`agentlib/tools/session_control.py`** — validation and blocklist
- **`agentlib/session.py`** — `_execute_slash_command_line_for_tool` (runs `execute_line` inside **`sink_capture_scope`**)
- **`agentlib/sink.py`** — `sink_capture_scope` merges printed REPL text into the tool result

The tool result is the command’s **full printed output** plus any structured `SessionLineResult.output` (deduped). The model should **read that text** and answer from it—not guess session state.

---

## Examples

| User question | Typical `command` |
|---------------|-------------------|
| Is thinking on? | `/set thinking show` |
| Which models are installed? | `/show models` |
| What LLM is primary? | `/show model` |
| List tools / toolsets | `/set tools list` |
| Inspect prefs group | `/set agent show` · `/set ollama show` |
| Turn thinking off | `/set thinking off` |
| Enable second opinion | `/set enable second_opinion` |
| Command reference | `/help` |
| Last token usage | `/usage` |

Progress in the TUI/CLI still shows lines like `→ [native] session_command Tool: session_command command='/set thinking show'`.

---

## Allowed vs blocked

**Allowed:** Most single-line slash commands: `/set`, `/show`, `/help`, `/clear`, `/cd`, `/context`, `/mcp`, `/compact`, `/usage`, `/set save`, `/set model …`, etc.

**Blocked** (user must run manually; tool returns an error string):

| Pattern | Reason |
|---------|--------|
| `/quit`, `/exit`, `/q` | Ends the session |
| `/while` | Unbounded agent loop |
| `/skill`, `/use-skill`, … | Multi-step agent workflows |
| `/send`, `/fork`, `/fork_background` | Multi-agent side effects |
| `/call_python`, `/run_command` | Arbitrary code / shell |
| `/source`, `/import` | Batch or inject arbitrary lines |
| `/set lock` | Permanent session freeze |
| `!command` | Shell escape |

When **`/set lock`** is active, mutating `/set` lines are rejected by the session (read-only **`show`** / **`list`** still work); the captured error text is returned to the model.

---

## Distinctions the model should respect

- **`thinking`** (Ollama `think` field) is **not** the same as casual reasoning prose in an answer.
- **`stream_thinking`** controls whether **[Thinking]** blocks appear in the UI when thinking is on.
- Use **`/set thinking show`** for status; do not use **`search_web`** to discover local Ollama models (use **`/show models`**).

---

## Related docs

- [Settings (`/set`)](settings-repl.md) — full topic map; **`/set help`** in the REPL is authoritative
- [REPL session & context](repl-session.md) — `/cd`, `/context`, `/source` (note: `/source` is blocked for `session_command`)
- [Core tools](core-tools.md) — tool table including `session_command`
- [README — Tools](../README.md#tools)
- **`agent_knowledge.txt`** — chat-assistant guidance (also mention in system context when automating)

**Do not** use the **`call_python`** tool to run `ai("/set …")` — `ai` is only defined for the **`/call_python`** slash command, not for in-turn tool calls.
