"""``/turn`` waits for a busy lane instead of failing fast."""

from __future__ import annotations

import threading
import time

from agentlib.tui_busy import is_lane_worker_thread, wait_until_lane_idle


def test_wait_until_lane_idle_blocks_then_proceeds():
    busy = {0}

    def is_busy(ln: int) -> bool:
        return ln in busy

    done = threading.Event()

    def waiter():
        wait_until_lane_idle(0, is_busy=is_busy, lane_worker_threads={}, poll=0.01)
        done.set()

    th = threading.Thread(target=waiter, daemon=True)
    th.start()
    time.sleep(0.05)
    assert not done.is_set()
    busy.discard(0)
    done.wait(timeout=1.0)
    assert done.is_set()
    th.join(timeout=1.0)


def test_wait_until_lane_idle_noop_on_lane_worker():
    busy = {0}
    workers = {0: threading.current_thread()}

    def is_busy(ln: int) -> bool:
        raise AssertionError("should not poll when nested on lane worker")

    wait_until_lane_idle(0, is_busy=is_busy, lane_worker_threads=workers)
    assert is_lane_worker_thread(workers, 0)
