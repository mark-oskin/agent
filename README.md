# agent

Local-first **CLI agent** that drives a **tool-using** loop against **Ollama** by default, with optional **hosted OpenAI-compatible** APIs for the primary model or a second opinion.

It can run as:

- **Interactive REPL** (default when you run it with no question)
- **One-shot mode** (pass a question on the command line; works without a TTY for scripting)

## What it does

- Sends your request (plus conversation history) to the model in a **structured JSON** protocol (`tool_call` / `answer`).
- **Core tools** cover web (DuckDuckGo search, fetch HTML), the filesystem, `git`, shell commands, and in-process Python.
- **Plugin toolsets** add optional tools from a `tools/` directory, gated by your settings and per-request routing.
- **Skills** (JSON in `skills/`) specialize behavior with extra prompt text, optional **multi-step workflows**, and a tighter **tool allowlist** when a skill matches.
- A **context window** helper can summarize older turns when the transcript grows; settings live in `~/.agent.json` and the REPL.
<!-- Embedding / Python API temporarily removed; CLI/REPL only. -->

## Install

The project uses [`uv`](https://github.com/astral-sh/uv).

### Develop / run from a clone

```bash
brew install uv   # or install uv by another method
uv sync
```

Use `./agent.py` or `uv run python agent.py` (see **Run** below).

### Put `agent` on your `PATH` (`uv tool install`)

From the repository root, install an isolated tool environment with the `agent` console script:

```bash
uv tool install .
```

Then use the **`agent`** command anywhere (same flags as `./agent.py`). **`uv`** prints where executables are linked (often `~/.local/bin`; ensure that directory is on your `PATH`).

Updates after pulling git changes:

```bash
uv tool install --reinstall .
```

Remove the tool:

```bash
uv tool uninstall agent
```

When this package is published to an index, you will also be able to run `uv tool install agent` (the PyPI/name story may evolve; installing from a path remains the reliable option for this repo).

## Run

| Mode | Command |
|------|---------|
| Interactive REPL | `./agent.py` · `uv run python agent.py` · or **`agent`** after `uv tool install .` |
| One-shot | `./agent.py "…"` · or **`agent` "…"` |
| Help | `./agent.py --help` · or **`agent --help`** |
| List tools and on/off state | `./agent.py --list-tools` · or **`agent --list-tools`** |

## CLI options (summary)

| Option | Purpose |
|--------|---------|
| `--config <file>` | Use this JSON file instead of `~/.agent.json` |
| `--model` / `--model=<name>` | Set primary model for this process (`OLLAMA_MODEL` for Ollama, or hosted model id if primary is hosted) |
| `--list-tools` | Print **core** tools, ids, and on/off for this session, then exit (for plugin toolsets, use `/settings tools` in the REPL) |
| `-enable-tool <id>` / `-disable-tool <id>` | Toggle tools for this run (repeatable) |
| `-verbose` [0\|1\|2] | `0` default; `1` logs tool use; `2` also streams raw model JSON (local Ollama) |
| `--prompt-template <name>` | Use a named prompt template for this run |
| `--load-context <file>` | Load session JSON; **requires** a question on the same command line |
| `--save-context <file>` | After the run, write the context bundle to this file |
| `--second-opinion` | Honor second-opinion when the model asks (see prefs) |
| `--cloud-ai` | Allow hosted / cloud-style backends per preferences |

Run `./agent.py --help` for the full text.

## Configuration

- **Default file:** `~/.agent.json` (versioned schema; the agent migrates/reads what it understands).
- **Override path:** `--config ./path/to/agent.json`
- In the **REPL**, use `/settings ollama`, `/settings openai`, `/settings agent` to view or set persisted settings, then **`/settings save`** to write the file.

## Interactive REPL (important commands)

| Command | Purpose |
|---------|---------|
| `/help` | Command list |
| `/quit` | Exit |
| `/clear` | Clear in-memory messages (and the stored skill for `/skill reuse`) |
| `/models` | List local Ollama models (`/api/tags`) |
| `/usage` | Show last Ollama prompt/completion usage from `/api/chat` |
| `/show model` | Current **primary** LLM (Ollama or hosted) |
| `/show reviewer` | Current **second-opinion** reviewer model |
| `/while [--max N] 'condition' do 'prompt' [, 'prompt' …]` | Same idea as **`while (condition) { … }`**: judge **1** = condition **true** (stay in loop), **0** = **false** (exit). After each **true** check, runs every **comma-separated** body prompt as its own REPL turn (step 1/N …), then re-checks. Capped by **--max** (default 50). See `/while help`. |
| `/skill list` | List available skills (ids) from the current `skills_dir`. |
| `/skill <skill> <request>` | Run a **specific** skill id (must exist in `skills_dir`); no model selection is performed. |
| `/skill auto <request>` | Ask the model to pick a skill, then run it (may be multi-step). |
| `/skill reuse <request>` | Follow-up using the same skill as the last `/skill` command (no re-selection). |
| `/settings …` | Model routing, tools, toolsets, thinking, system prompt, templates, context manager, `save`, etc. |

At `verbose=0`, startup is minimal (`Interactive mode. Type /help for commands.`). Use `/settings verbose 1` or `2` for a richer startup banner and tool logging.

## Embedding / Python API

Temporarily unavailable. Use the CLI/REPL for now.

## Core behavior

### Primary and reviewer LLMs

- **Primary** defaults to local **Ollama**. You can switch to a **hosted** OpenAI-compatible base URL from the REPL: `/settings primary llm hosted …` or `… ollama`.
- A **second-opinion** path can use another Ollama model or a separate hosted profile (`/settings second_opinion llm …`). Enable the feature with `/settings enable second_opinion` (and CLI flags as needed; see preferences).

### Context window

The agent can **compact** older transcript turns when a heuristic size threshold is hit. Configure with `/settings context …` (tokens, trigger/target fractions, `keep_tail`, on/off) and persist with `/settings save`. Some values can be overridden with `AGENT_CONTEXT_*` / `AGENT_DISABLE_CONTEXT_MANAGER` (see `/help environment`).

### Web search

Web search result count is capped; configure via `~/.agent.json` / `/settings agent` where exposed.

## Tools

### Built-in (core) tools

These are always defined by the agent; you can **disable** ones you do not want the model to call (blocked calls return an error string to the model so it can recover).

| ID | Role |
|----|------|
| `search_web` | Web search (DuckDuckGo HTML) |
| `fetch_page` | Fetch a URL (HTTP GET) |
| `run_command` | Run a shell command |
| `use_git` | Vetted `git` operations |
| `write_file` / `read_file` | Write or read a file |
| `list_directory` | List a directory |
| `download_file` | Download a URL to a path |
| `tail_file` | Read the end of a file |
| `replace_text` | Search-and-replace in a file |
| `call_python` | Run Python in-process |

Aliases exist for natural phrasing (e.g. “web search” → `search_web`); the REPL suggests names if you mistype one.

**REPL:** `/settings tools` lists core tools and **plugin** toolsets. `/settings enable …` / `/settings disable …` accept a tool id or a phrase.

**CLI:** `-enable-tool` / `-disable-tool` (same ids).

### Plugin toolsets (optional)

Extra tools ship as **Python modules** in a **tools directory**:

- **Default directory:** the repo’s `tools/` package (bundled `dev`, `desktop`, etc.).
- **Override:** `AGENT_TOOLS_DIR`, or the `tools_dir` field in `~/.agent.json` (set via session + `/settings save`).

Each plugin file defines a **`TOOLSET`** dict: `name`, `description`, optional `triggers` (keywords or `regex:…` patterns), and a list of tools (`id`, `description`, `aliases`, `handler`).

- **Enabling:** `/settings tools enable <toolset>` turns on a toolset and its tools. Toolsets are **off** until you enable them.
- **Routing:** If **multiple** toolsets are enabled, the agent tries to **expose only** toolsets whose triggers match the current user request (keeps the tool list smaller). If only **one** toolset is enabled, it is always considered active for that request. If nothing matches, enabled toolsets are all considered (fallback).
- **Reload:** after adding a new `.py` file, run `/settings tools reload` so imports refresh without restarting.
- **Details:** `/settings tools describe <tool-id>` or `<toolset-name>` for parameters and return shape (where documented).

### Example bundled toolsets

| Toolset | Notes |
|---------|--------|
| `dev` | e.g. `run_pytest` — runs the project test suite (`uv run pytest` when `uv.lock` exists) |
| `desktop` | e.g. `open_url` — macOS `open` in the default browser |

## Skills

Skills are **JSON files** (one file per skill) under **`skills/`** by default. Override the directory with **`AGENT_SKILLS_DIR`** or the `skills_dir` preference (via `/settings agent` and `/settings save`).

A typical skill object includes:

| Field | Purpose |
|-------|---------|
| `description` | Human-readable summary |
| `triggers` | Words/phrases; used to **match** the user’s message to a skill |
| `tools` | Subset of tool **ids** the skill allows when active |
| `prompt` | Extra system-style instructions injected when the skill runs |
| `workflow` (optional) | `max_steps`, `planner_prompt`, `step_prompt` for a **multi-step** run |

**REPL flow:**

- `/skill auto <request>` — asks the model to **pick** a skill, then runs the skill pipeline (including optional multi-step plans).
- `/skill <skill> <request>` — runs a specific skill id (no selection).
- `/skill reuse <request>` — reuses the **last** selected skill id without re-running selection (good for follow-ups).

One-shot mode can **auto-match** a skill from the user text using the same trigger machinery.

**Persisting** prompt templates, skills directory, and other session fields: `/settings save` writes `~/.agent.json` (or your `--config` file).

## Ollama “thinking” (optional)

For models that support it, the agent can send a **`think`** field and optionally **stream** thinking to the terminal:

- `/settings thinking on|off` and `level low|medium|high`
- `/settings enable stream_thinking` (or disable)

Some model families (e.g. `gpt-oss:*`) expect a **level**; if you enable thinking without a level, the agent may default the level to `"medium"`.

## Safety and robustness

- **Disabled tools** are not executed; the model gets a clear refusal string.
- **Tool failures** (exceptions, bad subprocess exit, etc.) are turned into a **result string** for the model instead of crashing the process.
- **Ctrl-C** during a long model or tool run cancels the **current** operation; the REPL **stays** running.
- In the REPL, **line editing** can use `readline` with history in `~/.agent_repl_history` (see verbose banner for details).

## Tests

```bash
uv run pytest
```

Some tests are marked for optional live Ollama runs; see `pyproject.toml` markers.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
