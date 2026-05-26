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
| Ask this agent a question (blocking) | `/turn self What is 2+2?` |
| Ask after current turn (async) | `/send self follow-up question` |

Progress in the TUI/CLI still shows lines like `→ [native] session_command Tool: session_command command='/set thinking show'`.

---

## Allowed vs blocked

**Allowed:** Most single-line REPL lines: slash commands (`/set`, `/show`, `/help`, `/send`, `/turn`, `/fork`, …) and shell escapes (`! cmd`, same as the REPL).

**Multi-agent (TUI):** Use **`/fork NAME`** to create a new lane (e.g. `command="/fork worker"` when the user wants an agent named worker). Use **`/list`** to see names. **`agent_send`** only messages lanes that already exist; it does not fork.

**`/send` vs `/turn`:** Both accept an agent name; **`self`** is the current session. **`/turn`** is **fully blocking**: it waits for the target lane to finish any in-flight work (TUI), runs the command with streaming UI updates, and **returns the full reply** in command output (so `session_command` with `/turn self …` gets the answer text). When `/turn self` runs **during** an agent turn (typical `session_command` path), it executes **in-process** on the same session (no extra TUI “You” line) and the nested turn **cannot** call `session_command` again (prevents `/turn self` loops). **`/send`** is **async** — during an in-flight agent turn it is **queued** until that turn finishes; otherwise it enqueues on another lane (TUI) or starts in the background (`self` on plain CLI).

**Blocked** (tool returns an error string):

| Pattern | Reason |
|---------|--------|
| `/quit`, `/exit`, `/q` | Ends the session |
| `/while` | Unbounded agent loop |
| `/skill`, `/use-skill`, `/use-skills`, `/reuse-skill` | Multi-step agent workflows |

When **`/set lock`** is active, mutating `/set` lines are rejected by the session (read-only **`show`** / **`list`** still work); the captured error text is returned to the model.

---

## Distinctions the model should respect

- **`thinking`** (Ollama `think` field) is **not** the same as casual reasoning prose in an answer.
- **`stream_thinking`** controls whether **[Thinking]** blocks appear in the UI when thinking is on.
- Use **`/set thinking show`** for status; do not use **`search_web`** to discover local Ollama models (use **`/show models`**).

---

## Related docs

- [Settings (`/set`)](settings-repl.md) — full topic map; **`/set help`** in the REPL is authoritative
- [REPL session & context](repl-session.md) — `/cd`, `/context`, `/source`, `/import`
- [Core tools](core-tools.md) — tool table including `session_command`
- [README — Tools](../README.md#tools)
- **`agent_knowledge.txt`** — chat-assistant guidance (also mention in system context when automating)

**Do not** use the **`call_python`** tool to run `ai("/set …")` — `ai` is only defined for the **`/call_python`** slash command, not for in-turn tool calls.
