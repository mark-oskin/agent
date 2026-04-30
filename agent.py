#!/usr/bin/env -S uv run python

"""
Entrypoint shim.

All application wiring lives in `agentlib.app`. This module remains as a stable
top-level executable for `./agent.py`.
"""

from agentlib.app import main


if __name__ == "__main__":
    main()

