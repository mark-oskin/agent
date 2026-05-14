# Developer guide

The repo ships a thin **`agent.py`** entrypoint; almost all behavior lives in **`agentlib/`** (CLI in `agentlib/app.py`, REPL loop in `agentlib/repl/loop.py`, session and slash commands in `agentlib/session.py`). Data directories include `skills/`, `prompt_templates/`, and optional plugin tools under `tools/`.

If you change behavior, update **`README.md`**, relevant **`docs/*.md`**, and this file when programmers need new facts.

**`agent_knowledge.txt`** (repository root) is a short, human/LLM-facing note on writing Python that runs **inside** `/call_python` and uses injected **`ai` / `session`** (same idea as `foreach_line.py`, `telegram_channel_ai.py`). It is listed in **`MANIFEST.in`** so source distributions include it. Update it when `/call_python` globals or return shapes change.

## Architecture overview

At a high level the agent runs a loop:

1. Build a **system instruction** (base prompt template + optional active skill prompt + runner/tool policy text).
2. Send chat messages to the model (Ollama `/api/chat` by default, or hosted equivalents).
3. Parse the assistant content as **agent JSON** with `action: "tool_call" | "answer"`.
4. If a tool is requested, run it, append the tool result back into the message list, and continue.

The main turn loop is implemented in:

- **`agentlib/runtime/turn.py`** — `run_agent_conversation_turn(...)` (orchestrates steps, tool execution, budgets).
- **`agentlib/app.py`** — `AgentApp` wires `ConversationTurnDeps` (includes `call_ollama_chat`, registry, paths).
- **`agentlib/llm/calls.py`** — `call_ollama_chat(...)` and related HTTP for Ollama / hosted profiles.

## Data directories

### Prompt templates

- Stored as JSON in `prompt_templates/`
- Selected by `--prompt-template` or `/settings prompt_template ...`
- Resolved and loaded via helpers used from `agentlib/app.py` (search `_resolved_prompt_templates_dir`, `_load_prompt_templates_from_dir`).

### Skills

Skills are JSON files under `skills/` (or prefs `agent.skills_dir` / legacy top-level `skills_dir`; see [Configuration & environment](environment.md)).

Each skill can define:

- `triggers`: phrases for matching a user request
- `tools`: tool allowlist when the skill is active
- `prompt`: extra instructions injected into the system prompt
- `workflow`: optional multi-step plan execution (`max_steps`, planner prompt, step prompt)

Key loaders / matchers live alongside the session / app (search `_load_skills_from_dir`, `_match_skill_detail`, `_ml_select_skill_id` for `/skill auto`).

## Tools

### Core tools

- **Catalog:** `agentlib/tools/routing.py` — `CORE_TOOL_ENTRIES` (ids, labels, aliases) and `CORE_TOOL_PROMPT_DOCS` (parameter text injected into the system prompt).
- **Implementation:** primarily `agentlib/tools/builtins.py` and related modules (e.g. web search helpers), dispatched through the tool execution path used from `run_agent_conversation_turn`.

The model can only call tools in the **effective tool allowlist** for that turn (core tools minus user-disabled tools plus routed plugin tools).

### Plugin toolsets

Plugin toolsets live in `tools/*.py` by default (or prefs `agent.tools_dir` / legacy top-level `tools_dir`; see [Configuration & environment](environment.md)).

Each plugin module exports:

```python
TOOLSET = {
  "name": "dev",
  "description": "...",
  "triggers": ["keyword", "regex:..."],   # optional
  "tools": [
    {
      "id": "run_pytest",
      "description": "...",
      "aliases": ("pytest", "run tests"),
      "handler": callable,
      # optional: "params": {...}, "returns": "...", "prompt_doc": "..."
    }
  ],
}
```

Loading and routing:

- **`agentlib/tools/plugins.py`** — `load_plugin_toolsets(...)` imports modules from the tools directory and registers handlers.
- **`agentlib/tools/registry.py`** — `ToolRegistry` exposes `load_plugin_toolsets`, `plugin_toolsets`, and `effective_enabled_tools_for_turn`.
- **`agentlib/tools/routing.py`** — `route_active_toolsets_for_request(...)`, `effective_enabled_tools_for_turn(...)` (trigger-based subset vs fallback).

## REPL commands

The interactive loop is **`agentlib.repl.loop.run_interactive_repl_loop`**, which drives **`agentlib.session.Session`** (`execute_line`, slash dispatch, context save/load hooks).

When adding or modifying commands:

- Ensure they remain discoverable via `/help`
- Add or update tests (e.g. `tests/test_to_100.py`, `tests/test_repl_extensions.py`, topic-specific files under `tests/`)
- For user-visible slash behavior, add or extend **`docs/`** (e.g. [while-repl.md](docs/while-repl.md), [extension-settings.md](docs/extension-settings.md), [settings-repl.md](docs/settings-repl.md), [help-repl.md](docs/help-repl.md), [environment.md](docs/environment.md))

### Multi-agent TUI (`agent_tui.py`)

- **Shortcuts** (`/send`, `/fork`, `/fork_background`, `/kill`) bypass the normal REPL runner and schedule host actions; transient feedback goes to **`_chat_logs`** (same transcript band as streamed slash output), not **`_activity_logs`** (thinking/tool telemetry).
- **Parsing** shares **`agentlib/tui_parse`** with the CLI: **`parse_fork_command`**, **`parse_send_command`** (comma-split inside optional `"..."`, with **`'…'`** and **`\,`** escapes), **`format_fork_command_line`**, **`format_send_command_line`**.
- **`/clipboard paste`** returns **`prefill_prompt`** on **`SessionLineResult`**; **`agent_tui`** injects into **`#prompt`**, **`repl/loop`** prints the snippet once for stdin users.

### Context I/O

Transcript load/save JSON helpers: **`agentlib/context/io.py`** (`load_context_messages`, `save_context_bundle`). Session methods delegate to these for `/context`, `/load_context`, `/save_context`, and post-turn auto-save when `session_save_path` is set.

### Embedding API (`import agent`)

Temporarily removed. `agent.py` is CLI/REPL-only for now.

## Packaging / install

The project defines a console script entry point in `pyproject.toml`:

- `agent = "agent:main"`

So users can do:

```bash
uv tool install .
agent --help
```

## Making changes safely (checklist)

- Update or add tests
- Run `uv run pytest`
- Update docs:
  - `README.md` (user-facing)
  - `docs/` pages when behavior is user-visible (see `docs/index.md`)
  - `DEVELOPERS.md` (programmer-facing)
