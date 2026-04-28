# Developer guide

This project is intentionally a **single-file CLI agent** (`agent.py`) plus data directories (`skills/`, `prompt_templates/`) and optional plugin tools (`tools/`). The goal is to keep iteration fast while still supporting extension.

If you change behavior, update **both** `README.md` and this file.

## Architecture overview

At a high level the agent runs a loop:

1. Build a **system instruction** (base prompt template + optional active skill prompt + runner/tool policy text).
2. Send chat messages to the model (Ollama `/api/chat` by default).
3. Parse the assistant content as **agent JSON** with `action: "tool_call" | "answer"`.
4. If a tool is requested, run it, append the tool result back into the message list, and continue.

The main loop is implemented in:

- `agent._run_agent_conversation_turn(...)`
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

## Embedding API (`import agent`)

### `AgentSession`

`AgentSession` provides a stateful interface for other Python programs:

- `session.messages`: conversation history for that session
- `session.run_query(text)`: run a normal agent turn
- `session.execute(line)`: run a REPL-style command line (e.g. `/settings ...`, `/while ...`)

Important limitation (current design):

- Some toggles are still backed by **process environment variables** (`AGENT_*`, `OLLAMA_*`). That means multiple sessions are best used **sequentially** unless you avoid env-backed settings.

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

