"""Lane busy-wait helpers for TUI blocking ``/turn``."""

from __future__ import annotations

import time
import threading
from typing import Callable


def is_lane_worker_thread(
    lane_worker_threads: dict[int, threading.Thread], lane_idx: int
) -> bool:
    th = lane_worker_threads.get(lane_idx)
    return th is not None and th is threading.current_thread()


def wait_until_lane_idle(
    lane_idx: int,
    *,
    is_busy: Callable[[int], bool],
    lane_worker_threads: dict[int, threading.Thread],
    poll: float = 0.05,
) -> None:
    """Block until lane ``lane_idx`` is idle. No-op on that lane's worker (nested turn)."""
    if is_lane_worker_thread(lane_worker_threads, lane_idx):
        return
    while is_busy(lane_idx):
        time.sleep(poll)
