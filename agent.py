#!/usr/bin/env python3

"""
Entrypoint shim.

All application wiring lives in `agentlib.app`. This module remains as a stable
top-level executable for `./agent.py`.
"""

import _bootstrap

_bootstrap.ensure_runtime(__file__)

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 v2 only supports OpenSSL 1\.1\.1\+.*LibreSSL.*",
    category=Warning,
    module=r"urllib3(\..*)?",
)

from agentlib.app import main


if __name__ == "__main__":
    main()

