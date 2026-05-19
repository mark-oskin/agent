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
- **`agent_knowledge.txt`** (repo root) is optional reading for **LLMs** writing Python that **drives this agent** from `/call_python` (`ai()`, `session`, batching and bridge patterns like `foreach_line.py` / `telegram_channel_ai.py`).
<!-- Embedding / Python API temporarily removed; CLI/REPL only. -->

## Documentation

Longer user guides (core tools, plugin toolsets, `/load` extensions, cwd/context/`/source`, `/while`, `/set` topics, extension prefs, **`/help`**, environment/prefs, queue, code pipeline) live under **[docs/](docs/index.md)**.

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
| Multi-agent TUI | Install the **tui** extra, then `uv run --extra tui python agent_tui.py` (see **Multi-agent Textual UI** below) |

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

## Multi-agent Textual UI (`agent_tui.py`)

Side-by-side agents in the terminal: pick a lane in the sidebar, type in the shared prompt, stream thinking and tool output separately from the transcript.

**Requires** the Textual stack (not installed by default):

```bash
uv sync --extra tui
uv run --extra tui python agent_tui.py
uv run --extra tui python agent_tui.py --agent Planner:llama3.2:latest --agent Coder:qwen2.5-coder:latest
```

Repeat **`--agent LABEL` or `--agent LABEL:MODEL`** for more lanes. Omit **`:MODEL`** to use your default primary model from prefs.

| Command | Purpose |
|---------|---------|
| **`/list`** · **`/switch NAME`** | Inspect agents and focus another lane |
| **`/fork NAME ["cmd1,cmd2,…"]`** | New lane; optional quoted list runs initial commands (commas split; use `'…'` or **`\,`** inside the quotes for a literal comma—the same rules as **`/fork help`**) |
| **`/fork_background …`** | Same as **`/fork`** but keep the current lane focused |
| **`/kill NAME`** | Remove a lane by display name (**one** lane must stay) |
| **`/send AGENT WORDS…`** | Run one prompt or slash line on another lane without blocking this one |
| **`/send AGENT "a,b"`** | Run **several** lines on another lane (`a` then `b`); inner quoting matches **`/fork`** (single-quoted fragments, **`\,`** for literal commas) |
| **`/clipboard`** | **`copy`** last assistant reply · **`copy all`** session JSON · **`paste`** loads the OS clipboard into the **prompt for editing** (nothing runs until you press Enter) |

In the TUI, normal slash-command help and confirmations appear in the **main transcript** (with your **You/**assistant messages), so they stay visually separate from the tool/thinking strip.

## Configuration

- **Default file:** `~/.agent.json` (versioned schema; the agent migrates/reads what it understands).
- **Override path:** `--config ./path/to/agent.json`
- In the **REPL**, use `/settings ollama`, `/settings openai`, `/settings agent` to view or set persisted settings, then **`/settings save`** to write the file.
- **`/settings`** is the same as **`/set`**. For a topic map (tools, primary LLM, context manager, templates, …), see **[docs/settings-repl.md](docs/settings-repl.md)**. Prefs vs env vars: **[docs/environment.md](docs/environment.md)**.

## Interactive REPL (important commands)

| Command | Purpose |
|---------|---------|
| `/help` | Command list (many topics use **`<topic> help`**, e.g. **`/set help`**, **`/while help`**) — see **[docs/help-repl.md](docs/help-repl.md)** |
| `/load` · `/unload` · `/extensions` | Load optional REPL extensions from a `.py` file (`register_repl`), list loaded paths, or unload all — see **[docs/repl-extensions.md](docs/repl-extensions.md)** |
| `/cd` · `/chdir` | Set the session working directory (tools, `run_command`, `!`) — see **[docs/repl-session.md](docs/repl-session.md)** |
| `/context …` · `/load_context` · `/save_context` | Load/save transcript JSON; **`/context start_log`** enables per-turn auto-save to a file — see **[docs/repl-session.md](docs/repl-session.md)** |
| `/source FILE` | Run each non-empty line of a UTF-8 file as a REPL line (batch setup) — **[docs/repl-session.md](docs/repl-session.md)** |
| `/quit` | Exit |
| `/clear` | Clear in-memory messages (and the stored skill for `/skill reuse`) |
| `/compact` | Optional target **N%** (e.g. `25%`) or **word count** (e.g. `400`); default **10%** of current estimated tokens. Asks the primary LLM to compress the transcript, then replaces all messages with that summary (same scratch reset as `/clear`). |
| `/show models` | List local Ollama models (`/api/tags`; alias **`/show local_models`**) |
| `/usage` · `/tokens` | Show last Ollama prompt/completion usage from `/api/chat` (same command; **`/tokens`** is an alias) |
| `/show model` | Current **primary** LLM (Ollama or hosted) |
| `/show reviewer` | Current **second-opinion** reviewer model |
| `/while [--max N] 'condition' do 'prompt' [, 'prompt' …]` | Same idea as **`while (condition) { … }`**: judge **1** = condition **true** (stay in loop), **0** = **false** (exit). After each **true** check, runs every **comma-separated** body prompt as its own REPL turn (step 1/N …), then re-checks. Capped by **--max** (default 50). See **`/while help`** and **[docs/while-repl.md](docs/while-repl.md)**. |
| `/skill list` | List available skills (ids) from the current `skills_dir`. |
| `/skill <skill> <request>` | Run a **specific** skill id (must exist in `skills_dir`); no model selection is performed. |
| `/skill auto <request>` | Ask the model to pick a skill, then run it (may be multi-step). |
| `/skill reuse <request>` | Follow-up using the same skill as the last `/skill` command (no re-selection). |
| `/settings …` | Model routing, tools, toolsets, thinking, system prompt, templates, context manager, `save`, etc. |
| `/clipboard copy` · `/clipboard copy all` · `/clipboard paste` | Clipboard: last answer, full session JSON, or load clipboard **into your next input** without auto-running (**paste** echoes in the stdin REPL; **`agent_tui`** puts text in the prompt box) |

**Multi-agent hooks** (**`/send`** and related) need a UI that wires enqueue/delegate (e.g. **`agent_tui.py`**); in a plain stdin REPL they print a setup hint instead of forwarding.

At `verbose=0`, startup is minimal (`Interactive mode. Type /help for commands.`). Use `/settings verbose 1` or `2` for a richer startup banner and tool logging.

## Embedding / Python API

Temporarily unavailable. Use the CLI/REPL for now.

## Core behavior

### Primary and reviewer LLMs

- **Primary** defaults to local **Ollama**. You can switch to a **hosted** OpenAI-compatible base URL from the REPL: `/settings primary llm hosted …` or `… ollama`.
- A **second-opinion** path can use another Ollama model or a separate hosted profile (`/settings second_opinion llm …`). Enable the feature with `/settings enable second_opinion` (and CLI flags as needed; see preferences).

### Context window

The agent can **compact** older transcript turns when a heuristic size threshold is hit. Configure with `/settings context …` (tokens, trigger/target fractions, `keep_tail`, on/off) and persist with `/settings save`. Compaction can also be turned off with **`agent.disable_context_manager`** in prefs (`/set agent set disable_context_manager true` then `/set save`). Details: **[docs/environment.md](docs/environment.md)**.

### Web search

Web search result count is capped; configure via `~/.agent.json` / `/settings agent` where exposed. **Backend:** set **`agent.search_web_backend`** to **`ddg`** (default, DuckDuckGo HTML + instant answer), **`searxng`** (needs **`agent.searxng_url`**), or **`brave`** (Brave Search API — set **`agent.brave_search_api_key`**). See **[docs/environment.md](docs/environment.md)**.

### MCP (Model Context Protocol) tools

Optional **external tools** via MCP servers:

- Use **`/mcp help`** in the REPL for **`list`**, **`status`**, **`add`**, **`remove`**, **`enable`** / **`disable`**, and **`reload`** (prefs keys **`agent.mcp_enabled`** and **`agent.mcp_servers`**). Full behavior, **stdio framing** (**`content-length`** vs **`ndjson`**), and examples: **[docs/environment.md](docs/environment.md#mcp-model-context-protocol)**.
- **stdio:** subprocess + JSON-RPC. Default wire format is **Content-Length** framing (typical for **`npx`** / Node servers). Python MCP SDK / **FastMCP** servers need **`--framing ndjson`** (stored as **`stdio_framing`** in prefs).
- **HTTP:** JSON-RPC POST to a URL (simple request/response).
- Each MCP tool appears as **`mcp_<server>_<tool>`** when it is in **this session's** **`enabled_tools`** (shared servers via **`/mcp enable`**, per-session via **`/mcp session on`**; **`/fork`** copies the parent's set).
- After MCP-related prefs change in the REPL, resync runs **in the background** when MCP is enabled; use **`/mcp status`** or **`/set tools reload`** as documented in **environment.md**.

## Tools

### Built-in (core) tools

These are always defined by the agent; you can **disable** ones you do not want the model to call (blocked calls return an error string to the model so it can recover).

| ID | Role |
|----|------|
| `search_web` | Web search (DuckDuckGo HTML) |
| `search_web_fetch_top` | Web search plus fetched excerpts from top result pages |
| `fetch_page` | Fetch a URL (HTTP GET) |
| `run_command` | Run a shell command |
| `use_git` | Vetted `git` operations |
| `write_file` / `read_file` | Write or read a file |
| `grep` | Regex search in a file or directory (ripgrep-like, Python `re`) |
| `list_directory` | List a directory |
| `download_file` | Download a URL to a path |
| `tail_file` | Read the end of a file |
| `replace_text` | Search-and-replace in a file |
| `call_python` | Run Python in-process |

Aliases exist for natural phrasing (e.g. “web search” → `search_web`); the REPL suggests names if you mistype one. Full table and parameter lines: **[docs/core-tools.md](docs/core-tools.md)**.

**REPL:** `/settings tools` lists core tools and **plugin** toolsets. `/settings enable …` / `/settings disable …` accept a tool id or a phrase.

**CLI:** `-enable-tool` / `-disable-tool` (same ids).

### Plugin toolsets (optional)

Extra tools ship as **Python modules** in a **tools directory**:

- **Default directory:** the repo’s `tools/` package (bundled `dev`, `desktop`, etc.).
- **Override:** the **`tools_dir`** field in `~/.agent.json` (set via `/set agent set tools_dir …` and `/set save`, or a top-level `tools_dir` in older prefs — see **[docs/environment.md](docs/environment.md)**).

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
| `browser` | Playwright browser tools (`browser_navigate`, …) — install **`uv sync --extra browser`**; see **[docs/plugin-toolsets.md](docs/plugin-toolsets.md)** |
| `applescript` | `run_applescript` (macOS `osascript`) — see **[docs/plugin-toolsets.md](docs/plugin-toolsets.md)** |
| `lanes` | `agent_send` for **agent_tui** cross-lane lines — see **[docs/plugin-toolsets.md](docs/plugin-toolsets.md)** |
| `queue` | Named in-memory FIFO lists for the model (`list_add`, …) — see **[docs/queue.md](docs/queue.md)** and optional **`/queue`** REPL extension |

## Skills

Skills are **JSON files** (one file per skill) under **`skills/`** by default. Override the directory with the **`skills_dir`** preference (`/set agent set skills_dir …` and `/set save`, or top-level `skills_dir` in older prefs — see **[docs/environment.md](docs/environment.md)**).

A typical skill object includes:

| Field | Purpose |
|-------|---------|
| `description` | Human-readable summary |
| `triggers` | Words/phrases; used to **match** the user’s message when trigger auto-match is enabled (see below) |
| `tools` | Subset of tool **ids** the skill allows when active |
| `prompt` | Extra system-style instructions injected when the skill runs |
| `workflow` (optional) | `max_steps`, `planner_prompt`, `step_prompt` for a **multi-step** run |

**REPL flow:**

- `/skill auto <request>` — asks the model to **pick** a skill, then runs the skill pipeline (including optional multi-step plans).
- `/skill <skill> <request>` — runs a specific skill id (no selection).
- `/skill reuse <request>` — reuses the **last** selected skill id without re-running selection (good for follow-ups).

Optional **trigger auto-match** on normal REPL turns: `/set agent set skill_auto_match_triggers true` then `/set save` (default **false**). When off, skills run only via `/skill …` commands.

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
