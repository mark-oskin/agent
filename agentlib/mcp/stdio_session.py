"""JSON-RPC MCP session over stdio (Content-Length or NDJSON framing)."""

from __future__ import annotations

import io
import json
import queue
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from agentlib.mcp.jsonrpc_compat import (
    MCP_PROTOCOL_VERSION_PREFERRED,
    jsonrpc_response_id_matches,
    mcp_initialize_params,
)
from agentlib.mcp.jsonrpc_framing import (
    read_framed_message,
    read_ndjson_message,
    write_framed_message,
    write_ndjson_message,
)


class StdioMcpSession:
    """One MCP server subprocess + bidirectional JSON-RPC."""

    def __init__(
        self,
        *,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        stdio_framing: str = "content-length",
    ):
        if not command:
            raise ValueError("stdio MCP server requires a non-empty command")
        self._cmd = list(command)
        norm = (stdio_framing or "content-length").strip().lower().replace("_", "-")
        if norm in ("jsonl", "newline"):
            norm = "ndjson"
        if norm not in ("content-length", "ndjson"):
            raise ValueError(f"stdio_framing must be 'content-length' or 'ndjson', not {stdio_framing!r}")
        self._stdio_framing = norm
        self._proc = subprocess.Popen(
            self._cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            bufsize=0,
        )
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._stdin: io.BufferedWriter = self._proc.stdin  # type: ignore[assignment]
        self._stdout: io.BufferedReader = self._proc.stdout  # type: ignore[assignment]
        self._stderr_fp = self._proc.stderr
        self._in_q: queue.Queue[Dict[str, Any]] = queue.Queue()
        self._err_q: queue.Queue[str] = queue.Queue()
        self._reader_stop = threading.Event()
        self._write_lock = threading.Lock()
        self._next_id = 1
        self._reader = threading.Thread(target=self._reader_loop, name="mcp-stdio-read", daemon=True)
        self._reader.start()
        self._drain_stderr()

    def _drain_stderr(self) -> None:
        if self._stderr_fp is None:
            return

        def run() -> None:
            try:
                for line in self._stderr_fp:
                    try:
                        self._err_q.put(line.decode("utf-8", errors="replace").rstrip())
                    except Exception:
                        pass
            except Exception:
                pass

        threading.Thread(target=run, name="mcp-stderr-drain", daemon=True).start()

    def _raise_if_proc_exited(self, method: str) -> None:
        """If the child has terminated, fail fast instead of waiting for a JSON-RPC timeout."""
        code = self._proc.poll()
        if code is None:
            return
        # Brief pause so the stderr drain thread can flush traceback lines.
        time.sleep(0.04)
        lines = self.drain_stderr_messages(max_items=48)
        tail = " | ".join(lines[-14:]) if lines else "(no stderr captured)"
        blob = "\n".join(lines).lower()
        hint = ""
        if ("modulenotfounderror" in blob and "mcp" in blob) or ("no module named" in blob and "mcp" in blob):
            hint = (
                " Hint: use the same Python that has the `mcp` SDK installed "
                "(project venv), e.g. `./.venv/bin/python3 -m your_pkg.server` — not bare `python` if it is system Python."
            )
        elif "content-length" in blob and "json_invalid" in blob:
            hint = (
                " Hint: Python MCP SDK / FastMCP stdio expects one JSON object per line (NDJSON), not Content-Length headers. "
                "Remove this server and re-add with: `/mcp add stdio NAME --framing ndjson ...`"
            )
        raise RuntimeError(
            f"MCP server process exited with code {code} before a JSON-RPC reply to {method!r}. Last stderr: {tail}{hint}"
        )

    def drain_stderr_messages(self, *, max_items: int = 20) -> List[str]:
        out: list[str] = []
        for _ in range(max_items):
            try:
                out.append(self._err_q.get_nowait())
            except queue.Empty:
                break
        return out

    def _write_jsonrpc(self, obj: Dict[str, Any]) -> None:
        with self._write_lock:
            if self._stdio_framing == "ndjson":
                write_ndjson_message(self._stdin, obj)
            else:
                write_framed_message(self._stdin, obj)

    def _reader_loop(self) -> None:
        try:
            while not self._reader_stop.is_set():
                try:
                    if self._stdio_framing == "ndjson":
                        msg = read_ndjson_message(self._stdout)
                    else:
                        msg = read_framed_message(self._stdout)
                except EOFError:
                    break
                except Exception as e:
                    self._in_q.put({"__fatal__": str(e)})
                    break
                self._in_q.put(msg)
        finally:
            pass

    def close(self) -> None:
        self._reader_stop.set()
        try:
            self._stdin.close()
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass

    def _send_notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._write_jsonrpc(payload)

    def request(self, method: str, params: Optional[Dict[str, Any]] = None, *, timeout_s: float = 120.0) -> Any:
        req_id = self._next_id
        self._next_id += 1
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            payload["params"] = params
        self._write_jsonrpc(payload)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self._raise_if_proc_exited(method)
            remaining = max(0.05, deadline - time.time())
            try:
                msg = self._in_q.get(timeout=min(0.5, remaining))
            except queue.Empty:
                self._raise_if_proc_exited(method)
                continue
            if "__fatal__" in msg:
                raise RuntimeError(str(msg.get("__fatal__")))
            if msg.get("method") == "notifications/message":
                continue
            if not jsonrpc_response_id_matches(msg.get("id"), req_id):
                continue
            if msg.get("error"):
                err = msg["error"]
                if isinstance(err, dict):
                    raise RuntimeError(str(err.get("message") or err))
                raise RuntimeError(str(err))
            return msg.get("result")
        self._raise_if_proc_exited(method)
        raise TimeoutError(f"MCP request timeout for method {method!r}")

    def handshake(self) -> None:
        self.request(
            "initialize",
            mcp_initialize_params(protocol_version=MCP_PROTOCOL_VERSION_PREFERRED),
            timeout_s=30.0,
        )
        self._send_notification("notifications/initialized", {})

    def list_tools(self) -> List[Dict[str, Any]]:
        result = self.request("tools/list", {}, timeout_s=30.0)
        if not isinstance(result, dict):
            return []
        raw = result.get("tools")
        return raw if isinstance(raw, list) else []

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        params: Dict[str, Any] = {"name": name}
        if arguments:
            params["arguments"] = arguments
        return self.request("tools/call", params, timeout_s=300.0)
