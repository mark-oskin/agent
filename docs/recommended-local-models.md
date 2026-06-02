---
title: Recommended local models
description: Ollama models that work well with the agent for local inference — starting points, not requirements.
---

# Recommended local models

The agent expects an [**Ollama**](https://ollama.com/) model when you run locally (unless you point prefs at a hosted primary model). Any compatible model can work; these models are **good defaults** we exercise often—balanced instruction following, tool-friendly behavior, and reasonable size on consumer GPUs.

Pull before first use:

```bash
ollama pull qwen3.6:latest
ollama pull gpt-oss:20b
ollama pull gemma4:26b
ollama pull laguna-xs.2:latest
```

## `qwen3.6:latest`

Strong general-purpose instruct model—solid default for **interactive REPL**, skills, and native tool turns. Matches the project’s default prefs hint (`ollama.model`) when you want a single “drop in and go” choice.

## `gpt-oss:20b`

Heavier **reasoning-oriented** option when you want more depth at the cost of VRAM and latency—useful for harder analysis or coding passes while staying fully local.

## `gemma4:26b` and `gemma4:31b`

**Gemma 4** is on the recommended list. **`gemma4:26b`** is a strong general-purpose local choice—good instruction following and a practical size for everyday REPL and tool use. **`gemma4:31b`** is the larger variant when you have the VRAM and want more capacity.

```bash
ollama pull gemma4:26b
# optional, heavier:
ollama pull gemma4:31b
```

## `laguna-xs.2:latest`

Lightweight instruct option for fast local iteration—useful for everyday interactive sessions, quick coding help, and lightweight tool turns. For workflows that need **tight integration with your local machine or desktop apps**, still expect some friction and test your specific setup.

## Models fine for many tasks—but weaker on niche tools

Models such as **Llama 3.2**, **LFM2**, and **GLM-4.7-flash** are widely usable for general chat, analysis, and **ordinary coding**. They often **do not follow unusual tool workflows reliably**—for example **deep integration with your machine**: long tool chains, strict JSON/tool contracts, or automation that depends on how *your* environment is set up.

The repo’s **[`local_knowledge.txt`](../local_knowledge.txt)** (at the repository root) is where **you** can place examples and notes about integrating with **your** specific local user environment—paths, apps, conventions, whatever matters on your box. **Agents do not load it automatically**; you must **tell the agent to read `local_knowledge.txt` and learn from it** if you want that guidance in play.

Some of these models **can** also be nudged toward difficult workflows with **heavy prompting** or narrower sessions, but expect friction compared with models that were tuned harder for tool calling; test your stack before relying on them for automation-heavy setups.

---

**See also:** [Settings (`/set`) — model routing](settings-repl.md) · [README — Configuration](../README.md#configuration)
