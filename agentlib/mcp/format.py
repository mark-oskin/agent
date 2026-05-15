"""Format MCP ``tools/call`` results as plain text for the LLM."""

from __future__ import annotations

import json
from typing import Any


def format_tool_result(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if not isinstance(result, dict):
        return str(result)
    if "structuredContent" in result:
        try:
            return json.dumps(result["structuredContent"], ensure_ascii=False)
        except Exception:
            return str(result["structuredContent"])
    raw_content = result.get("content")
    if isinstance(raw_content, list):
        parts: list[str] = []
        for block in raw_content:
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            if block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            else:
                parts.append(json.dumps(block, ensure_ascii=False))
        return "\n".join(p for p in parts if p).strip() or json.dumps(result, ensure_ascii=False)
    return json.dumps(result, ensure_ascii=False)
