# agent

Local-first CLI agent that talks to **Ollama** (default) and can optionally use a **hosted OpenAI-compatible** endpoint.

It runs as:
- **Interactive REPL** (default)
- **One-shot CLI** (pass a question on the command line)

## Install

This repo is set up for [`uv`](https://github.com/astral-sh/uv).

```bash
brew install uv
uv sync
```

## Run

Interactive:

```bash
./agent.py
```

One-shot:

```bash
./agent.py "summarize this repo"
```

Help:

```bash
./agent.py --help
```

## Requirements

- **Python**: 3.9+
- **Ollama** running locally (default): set `OLLAMA_HOST` if it is not `http://localhost:11434`

## Configuration

Settings are persisted in **`~/.agent.json`** and edited from the REPL:

- `/settings ollama show|keys|set|unset`
- `/settings openai show|keys|set|unset`
- `/settings agent show|keys|set|unset`
- `/settings save`

To use a different config file:

```bash
./agent.py --config ./my-agent.json
```

## Tools (enable/disable)

The agent has a small built-in tool set (web search, fetch URL, git, filesystem, shell, etc).

In the REPL:

- `/settings tools` shows tool ids and whether they’re enabled
- `/settings enable <tool or phrase>`
- `/settings disable <tool or phrase>`

Examples:

```text
/settings disable web search
/settings enable shell
```

In one-shot mode you can also gate tools from the command line:

```bash
./agent.py -disable-tool search_web "answer without using the web"
./agent.py -enable-tool run_command "run a command"
```

If a tool is disabled and the model tries to call it anyway, the tool call is **blocked** and the model receives a tool error string so it can recover.

## Skills

Skills are lightweight prompt “profiles” that help the model act like a specialist (e.g. debugging, docker, security audit) and optionally run a short workflow.

In the REPL:

- `/use-skills <request>` asks the model to pick a skill and then runs it
- `/reuse-skill <request>` reuses the previously selected skill for a follow-up

Skills live under `skills/` (JSON files). You can override the skills directory with:

- `AGENT_SKILLS_DIR` (environment/config) or `/settings agent set SKILLS_DIR <path>` then `/settings save`

### Thinking / streaming thinking (Ollama)

- `/settings thinking on|off`
- `/settings thinking level low|medium|high`
- `/settings enable stream_thinking` / `/settings disable stream_thinking`

Note: for `gpt-oss:*` models, `think` requires a level; if thinking is enabled without a level, the agent defaults it to `"medium"`.

## Running tests

```bash
uv run pytest
```

## License

MIT. See `LICENSE`.

