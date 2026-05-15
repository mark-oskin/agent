"""JSON-RPC MCP session over HTTP POST (simple request/response servers)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import requests

from agentlib.mcp.jsonrpc_compat import (
    MCP_PROTOCOL_VERSION_PREFERRED,
    jsonrpc_response_id_matches,
    mcp_initialize_params,
)


class HttpMcpSession:
    """Minimal MCP client: POST JSON-RPC to a fixed URL."""

    def __init__(self, url: str, *, headers: Optional[Dict[str, str]] = None, timeout_s: float = 120.0):
        self._url = (url or "").strip()
        if not self._url.startswith(("http://", "https://")):
            raise ValueError("HTTP MCP url must start with http:// or https://")
        self._headers = {"Content-Type": "application/json", **(headers or {})}
        self._timeout = timeout_s
        self._next_id = 1

    def close(self) -> None:
        return

    def drain_stderr_messages(self, *, max_items: int = 20) -> List[str]:
        return []

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(self._url, data=json.dumps(payload).encode("utf-8"), headers=self._headers, timeout=self._timeout)
        r.raise_for_status()
        ct = (r.headers.get("Content-Type") or "").lower()
        if "application/json" not in ct and r.content[:1] in (b"{", b"["):
            pass
        try:
            msg = r.json()
        except json.JSONDecodeError as e:
            raise ValueError(f"MCP HTTP response is not JSON: {e}") from e
        if not isinstance(msg, dict):
            raise ValueError("MCP HTTP JSON root must be an object")
        return msg

    def request(self, method: str, params: Optional[Dict[str, Any]] = None, *, timeout_s: float = 120.0) -> Any:
        req_id = self._next_id
        self._next_id += 1
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            payload["params"] = params
        prev = self._timeout
        try:
            self._timeout = timeout_s
            msg = self._post(payload)
        finally:
            self._timeout = prev
        if msg.get("error"):
            err = msg["error"]
            if isinstance(err, dict):
                raise RuntimeError(str(err.get("message") or err))
            raise RuntimeError(str(err))
        if not jsonrpc_response_id_matches(msg.get("id"), req_id):
            raise RuntimeError("MCP HTTP JSON-RPC id mismatch")
        return msg.get("result")

    def handshake(self) -> None:
        self.request(
            "initialize",
            mcp_initialize_params(protocol_version=MCP_PROTOCOL_VERSION_PREFERRED),
            timeout_s=60.0,
        )
        notify = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        try:
            self._post(notify)
        except Exception:
            pass

    def list_tools(self) -> List[Dict[str, Any]]:
        result = self.request("tools/list", {}, timeout_s=60.0)
        if not isinstance(result, dict):
            return []
        raw = result.get("tools")
        return raw if isinstance(raw, list) else []

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        params: Dict[str, Any] = {"name": name}
        if arguments:
            params["arguments"] = arguments
        return self.request("tools/call", params, timeout_s=300.0)
