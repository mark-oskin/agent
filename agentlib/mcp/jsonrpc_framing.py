"""MCP stdio transport framing: Content-Length + body (TypeScript-style) or NDJSON lines (Python SDK)."""

from __future__ import annotations

import json
from typing import Any, BinaryIO, Dict, Optional


def write_framed_message(fp: BinaryIO, obj: Dict[str, Any]) -> None:
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    fp.write(header + body)
    fp.flush()


def read_framed_message(fp: BinaryIO) -> Dict[str, Any]:
    """Read one MCP message from a binary stream."""
    header_lines: list[bytes] = []
    while True:
        line = fp.readline()
        if not line:
            raise EOFError("unexpected EOF while reading MCP headers")
        if line in (b"\r\n", b"\n"):
            break
        header_lines.append(line.rstrip(b"\r\n"))
    headers = b"\n".join(header_lines).decode("latin-1")
    cl: Optional[int] = None
    for raw in headers.split("\n"):
        if ":" not in raw:
            continue
        k, v = raw.split(":", 1)
        if k.strip().lower() == "content-length":
            try:
                cl = int(v.strip())
            except ValueError:
                cl = None
            break
    if cl is None or cl < 0:
        raise ValueError(f"missing or invalid Content-Length in headers: {headers!r}")
    body = fp.read(cl)
    if len(body) != cl:
        raise EOFError(f"expected {cl} bytes body, got {len(body)}")
    try:
        parsed = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON in MCP body: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError("MCP JSON-RPC root must be an object")
    return parsed


def write_ndjson_message(fp: BinaryIO, obj: Dict[str, Any]) -> None:
    """Write one JSON-RPC message as a single UTF-8 line (Python MCP SDK / FastMCP stdio style)."""
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    fp.write(line.encode("utf-8"))
    fp.flush()


def read_ndjson_message(fp: BinaryIO) -> Dict[str, Any]:
    """Read one JSON object line from an NDJSON MCP stream (skip blank lines)."""
    while True:
        raw = fp.readline()
        if not raw:
            raise EOFError("unexpected EOF reading NDJSON MCP message")
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON in NDJSON MCP line: {e}") from e
        if not isinstance(parsed, dict):
            raise ValueError("MCP JSON-RPC root must be an object")
        return parsed
