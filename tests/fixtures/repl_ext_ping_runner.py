"""Minimal script for ``/call_python`` subcommand tests (loaded via bridge extension)."""

from __future__ import annotations

import sys

if __name__ == "__call_python__":
    if len(sys.argv) >= 2 and sys.argv[1] == "ping":
        tail = " ".join(sys.argv[2:])
        print(f"pong:{tail}")
