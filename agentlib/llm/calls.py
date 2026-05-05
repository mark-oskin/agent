from __future__ import annotations

import json
from typing import AbstractSet, Callable, Optional, Tuple

import requests
from requests.exceptions import HTTPError

from agentlib.sink import sink_emit


_THINK_FALLBACK_WARNING = (
    "Ollama rejected thinking for this model (HTTP 400); retrying without the think option.\n"
)

def _emit_full_llm_prompts_if_verbose(messages: list, *, verbose: int, backend: str, model: str, format_json: bool) -> None:
    """
    Verbose level 3: emit the exact prompts (messages list) sent to the LLM.
    """
    if int(verbose) < 3:
        return
    try:
        header = f"[*] LLM request prompts (backend={backend}, model={model!r}, format_json={bool(format_json)}):"
        sink_emit({"type": "output", "text": header})
        sink_emit(
            {
                "type": "output",
                "text": json.dumps(messages, ensure_ascii=False, indent=2),
            }
        )
    except Exception:
        # Never fail the model call due to verbose printing.
        return


def _should_retry_ollama_chat_without_think(exc: BaseException, body: dict) -> bool:
    """If True, caller may retry the same /api/chat body with ``think: False``."""
    if not isinstance(exc, HTTPError) or exc.response is None:
        return False
    if exc.response.status_code != 400:
        return False
    return body.get("think") is not False


def call_ollama_plaintext(
    *,
    base_url: str,
    messages: list,
    model: str,
    think_value: object,
    merge_stream_message_chunks: Callable[..., Tuple[dict, Optional[dict], bool]],
) -> str:
    """Ollama /api/chat without JSON format — for second-opinion reviewer text."""
    base = (base_url or "").rstrip("/")
    url = f"{base}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": think_value,
    }

    def run_chat(streaming: bool) -> Tuple[str, Optional[dict]]:
        body = {**payload, "stream": streaming}
        for attempt in range(2):
            try:
                if streaming:
                    with requests.post(url, json=body, stream=True, timeout=600) as r:
                        r.raise_for_status()
                        msg, usage, _ = merge_stream_message_chunks(
                            r.iter_lines(decode_unicode=True), stream_chunks=False
                        )
                    return (msg.get("content") or "").strip(), usage
                r = requests.post(url, json=body, timeout=600)
                r.raise_for_status()
                data = r.json()
                msg = data.get("message") or {}
                usage = data.get("usage") if isinstance(data, dict) else None
                return (msg.get("content") or "").strip(), usage if isinstance(usage, dict) else None
            except HTTPError as e:
                if attempt == 0 and _should_retry_ollama_chat_without_think(e, body):
                    sink_emit({"type": "warning", "text": _THINK_FALLBACK_WARNING})
                    body = {**body, "think": False}
                    continue
                raise

    text, _usage = run_chat(streaming=True)
    if not text:
        text, _usage = run_chat(streaming=False)
    return text or "(empty reviewer response)"


def call_hosted_chat_plain(messages: list, *, base_url: str, model: str, api_key: str) -> str:
    """Non-streaming chat.completions for OpenAI-compatible APIs (OpenAI, Grok, Groq, Azure, etc.)."""
    key = (api_key or "").strip()
    if not key:
        return "Cloud AI error: api_key is not set."
    base = (base_url or "").rstrip("/")
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0.3,
    }
    try:
        r = requests.post(url, json=body, headers=headers, timeout=120)
        r.raise_for_status()
        data = r.json()
        choice0 = (data.get("choices") or [{}])[0]
        msg = choice0.get("message") or {}
        return (msg.get("content") or "").strip() or "(empty cloud response)"
    except Exception as e:
        return f"Cloud AI error: {e}"


def call_llm_json_content(
    messages: list,
    *,
    primary_profile,
    verbose: int,
    ollama_base_url: str,
    ollama_model: str,
    merge_stream_message_chunks: Callable[..., Tuple[dict, Optional[dict], bool]],
    ollama_usage_from_chat_response: Callable[[dict], Optional[dict]],
    set_last_ollama_usage: Callable[[Optional[dict]], None],
) -> str:
    """
    One-shot model call: return assistant *content* as stored by the model.
    Does NOT run agent post-processing, so the reply can be arbitrary JSON.
    """
    prof = primary_profile
    if prof is not None and getattr(prof, "backend", "") == "hosted":
        set_last_ollama_usage(None)
        key = (getattr(prof, "api_key", "") or "").strip()
        if not key:
            return json.dumps({"_call_error": "api_key is not set."})
        base = str(getattr(prof, "base_url", "") or "").rstrip("/")
        url = f"{base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": getattr(prof, "model", ""),
            "messages": messages,
            "stream": False,
            "temperature": 0.2,
        }
        _emit_full_llm_prompts_if_verbose(
            messages,
            verbose=verbose,
            backend="hosted",
            model=str(getattr(prof, "model", "") or ""),
            format_json=False,
        )
        try:
            r = requests.post(url, json=body, headers=headers, timeout=300)
            r.raise_for_status()
            data = r.json()
            choice0 = (data.get("choices") or [{}])[0]
            msg = choice0.get("message") or {}
            text = (msg.get("content") or "").strip()
            if verbose >= 2 and text:
                sink_emit({"type": "output", "text": text})
            return text or ""
        except Exception as e:
            return json.dumps({"_call_error": f"Hosted JSON call error: {e}"})

    set_last_ollama_usage(None)
    base = (ollama_base_url or "").rstrip("/")
    url = f"{base}/api/chat"
    payload = {
        "model": ollama_model,
        "messages": messages,
        "stream": True,
        "format": "json",
        "think": False,
    }
    _emit_full_llm_prompts_if_verbose(messages, verbose=verbose, backend="ollama", model=ollama_model, format_json=True)
    try:

        def run_once(streaming: bool) -> Tuple[str, Optional[dict]]:
            body = {**payload, "stream": streaming}
            if streaming:
                with requests.post(url, json=body, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    msg, usage, _ = merge_stream_message_chunks(
                        r.iter_lines(decode_unicode=True), stream_chunks=verbose >= 2
                    )
            else:
                r = requests.post(url, json=body, timeout=300)
                r.raise_for_status()
                data = r.json()
                msg = data.get("message") or {}
                usage = ollama_usage_from_chat_response(data)
            return ((msg.get("content") or "").strip(), usage)

        text, usage = run_once(streaming=True)
        if not text:
            text, usage = run_once(streaming=False)
        if usage:
            set_last_ollama_usage(usage)
        if verbose >= 2 and text:
            sink_emit({"type": "output", "text": "", "end": "\n"})
        return text
    except Exception as e:
        return json.dumps({"_call_error": f"Ollama JSON call error: {e}"})


def call_hosted_agent_chat(
    messages: list,
    *,
    base_url: str,
    model: str,
    api_key: str,
    enabled_tools: Optional[AbstractSet[str]],
    verbose: int,
    message_to_agent_json_text: Callable[[dict, Optional[AbstractSet[str]]], str],
    verbose_emit_final_agent_readable: Callable[[str], None],
) -> str:
    """Hosted primary agent: same JSON contract as Ollama /api/chat + format json."""
    key = (api_key or "").strip()
    if not key:
        return json.dumps({"action": "error", "error": "api_key is not set."})
    base = (base_url or "").rstrip("/")
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0.3,
    }
    _emit_full_llm_prompts_if_verbose(messages, verbose=verbose, backend="hosted", model=model, format_json=False)
    try:
        r = requests.post(url, json=body, headers=headers, timeout=600)
        r.raise_for_status()
        data = r.json()
        choice0 = (data.get("choices") or [{}])[0]
        msg = choice0.get("message") or {}
        text = message_to_agent_json_text(msg, enabled_tools).strip()
        if not text:
            return json.dumps({"action": "answer", "answer": "No response from model."})
        if verbose >= 2:
            sink_emit({"type": "output", "text": text})
            verbose_emit_final_agent_readable(text)
        return text
    except Exception as e:
        sink_emit({"type": "debug", "text": f"[DEBUG] Hosted chat error: {e}"})
        return json.dumps({"action": "error", "error": str(e)})


def call_ollama_chat(
    messages: list,
    *,
    primary_profile,
    enabled_tools: Optional[AbstractSet[str]],
    verbose: int,
    ollama_base_url: str,
    ollama_model: str,
    ollama_think_value: object,
    ollama_debug: bool,
    merge_stream_message_chunks: Callable[..., Tuple[dict, Optional[dict], bool]],
    ollama_usage_from_chat_response: Callable[[dict], Optional[dict]],
    message_to_agent_json_text: Callable[[dict, Optional[AbstractSet[str]]], str],
    verbose_emit_final_agent_readable: Callable[[str], None],
    format_ollama_usage_line: Callable[[dict], str],
    set_last_ollama_usage: Callable[[Optional[dict]], None],
    call_hosted_agent_chat_impl: Callable[..., str],
) -> str:
    """
    Agent chat: local Ollama JSON /api/chat, or hosted OpenAI-compatible chat.completions.
    """
    prof = primary_profile
    if prof is not None and getattr(prof, "backend", "") == "hosted":
        set_last_ollama_usage(None)
        return call_hosted_agent_chat_impl(
            messages,
            profile=prof,
            enabled_tools=enabled_tools,
            verbose=verbose,
        )

    base = (ollama_base_url or "").rstrip("/")
    url = f"{base}/api/chat"
    payload = {
        "model": ollama_model,
        "messages": messages,
        "stream": True,
        "format": "json",
        "think": ollama_think_value,
    }
    _emit_full_llm_prompts_if_verbose(messages, verbose=verbose, backend="ollama", model=ollama_model, format_json=True)
    stream_llm = verbose >= 2

    def run_chat(streaming: bool) -> Tuple[str, Optional[dict], bool]:
        body = {**payload, "stream": streaming}
        for attempt in range(2):
            try:
                if streaming:
                    with requests.post(url, json=body, stream=True, timeout=600) as r:
                        r.raise_for_status()
                        msg, usage, streamed = merge_stream_message_chunks(
                            r.iter_lines(decode_unicode=True), stream_chunks=stream_llm
                        )
                    if ollama_debug:
                        sink_emit({"type": "debug", "text": f"[DEBUG] Ollama merged message: {msg!r}"})
                    text = message_to_agent_json_text(msg, enabled_tools)
                    return text, usage, streamed
                r = requests.post(url, json=body, timeout=600)
                r.raise_for_status()
                data = r.json()
                if ollama_debug:
                    sink_emit({"type": "debug", "text": f"[DEBUG] Ollama API response: {data!r}"})
                msg = data.get("message") or {}
                text = message_to_agent_json_text(msg, enabled_tools)
                usage = ollama_usage_from_chat_response(data)
                if stream_llm and text.strip():
                    sink_emit({"type": "output", "text": text})
                    return text, usage, True
                return text, usage, False
            except HTTPError as e:
                if attempt == 0 and _should_retry_ollama_chat_without_think(e, body):
                    sink_emit({"type": "warning", "text": _THINK_FALLBACK_WARNING})
                    body = {**body, "think": False}
                    continue
                raise

    try:
        text, usage, streamed = run_chat(streaming=True)
        text = text.strip()
        if not text:
            text2, usage2, streamed2 = run_chat(streaming=False)
            text = text2.strip()
            if usage2:
                usage = usage2
            streamed = streamed or streamed2
        if usage:
            set_last_ollama_usage(usage)
        if stream_llm:
            if streamed:
                sink_emit({"type": "output", "text": "", "end": "\n"})
            if text:
                verbose_emit_final_agent_readable(text)
        if stream_llm and usage:
            sink_emit({"type": "output", "text": format_ollama_usage_line(usage)})
        if not text:
            return json.dumps({"action": "answer", "answer": "No response from model."})
        return text
    except Exception as e:
        sink_emit({"type": "debug", "text": f"[DEBUG] Request error: {e}"})
        return json.dumps({"action": "error", "error": str(e)})

