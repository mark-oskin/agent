# Developer guide

This project is intentionally a **single-file CLI agent** (`agent.py`) plus data directories (`skills/`, `prompt_templates/`) and optional plugin tools (`tools/`). The goal is to keep iteration fast while still supporting extension.

If you change behavior, update **both** `README.md` and this file.

**`agent_knowledge.txt`** (repository root) is a short, human/LLM-facing note on writing Python that runs **inside** `/call_python` and uses injected **`ai` / `session`** (same idea as `foreach_line.py`, `telegram_channel_ai.py`). It is listed in **`MANIFEST.in`** so source distributions include it. Update it when `/call_python` globals or return shapes change.

## Architecture overview

At a high level the agent runs a loop:

1. Build a **system instruction** (base prompt template + optional active skill prompt + runner/tool policy text).
2. Send chat messages to the model (Ollama `/api/chat` by default).
3. Parse the assistant content as **agent JSON** with `action: "tool_call" | "answer"`.
4. If a tool is requested, run it, append the tool result back into the message list, and continue.

The main loop is implemented in:

- `agentlib.runtime.run_agent_conversation_turn(...)` (wiring: `agent._conversation_turn_deps()`)
- `agent.call_ollama_chat(...)` (agent JSON via Ollama) and hosted equivalents

## Data directories

### Prompt templates

- Stored as JSON in `prompt_templates/`
- Selected by `--prompt-template` or `/settings prompt_template ...`
- Resolved via `_resolved_prompt_templates_dir(...)` and loaded by `_load_prompt_templates_from_dir(...)`

### Skills

Skills are JSON files under `skills/` (or `AGENT_SKILLS_DIR` / prefs `skills_dir`).

Each skill can define:

- `triggers`: phrases for matching a user request
- `tools`: tool allowlist when the skill is active
- `prompt`: extra instructions injected into the system prompt
- `workflow`: optional multi-step plan execution (`max_steps`, planner prompt, step prompt)

Key loaders / matchers:

- `_load_skills_from_dir(...)`
- `_match_skill_detail(...)`
- `_ml_select_skill_id(...)` (model-assisted selection for `/skill auto`)

## Tools

### Core tools

Core tools are implemented inside `agent.py` and named in `_CORE_TOOLS`.

The model can only call tools in the **effective tool allowlist** for that turn (core tools minus user-disabled tools plus routed plugin tools).

### Plugin toolsets

Plugin toolsets live in `tools/*.py` by default (or `AGENT_TOOLS_DIR` / prefs `tools_dir`).

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
      # optional: "params": {...}, "returns": "..."
    }
  ],
}
```

Loading and routing:

- `_load_plugin_toolsets(...)` imports `tools/*.py` and registers their handlers.
- `_route_active_toolsets_for_request(...)` selects a subset of enabled toolsets based on trigger matches.
- `_effective_enabled_tools_for_turn(...)` combines base tools + routed toolset tools.

## REPL commands

The interactive REPL lives in `_interactive_repl(...)`.

When adding or modifying commands:

- Ensure they remain discoverable via `/help`
- Add or update tests under `tests/test_to_100.py`

### Multi-agent TUI (`agent_tui.py`)

- **Shortcuts** (`/send`, `/fork`, `/fork_background`, `/kill`) bypass the normal REPL runner and schedule host actions; transient feedback goes to **`_chat_logs`** (same transcript band as streamed slash output), not **`_activity_logs`** (thinking/tool telemetry).
- **Parsing** shares **`agentlib/tui_parse`** with the CLI: **`parse_fork_command`**, **`parse_send_command`** (comma-split inside optional `"..."`, with **`'…'`** and **`\,`** escapes), **`format_fork_command_line`**, **`format_send_command_line`**.
- **`/clipboard paste`** returns **`prefill_prompt`** on **`SessionLineResult`**; **`agent_tui`** injects into **`#prompt`**, **`repl/loop`** prints the snippet once for stdin users.

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
  - `DEVELOPERS.md` (programmer-facing)

