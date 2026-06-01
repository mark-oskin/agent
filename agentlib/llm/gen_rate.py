"""Generation rate (tokens per second) for live UI feedback."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from agentlib.llm.token_estimate import CharsPerTokenEstimator, estimate_tokens_from_text

__all__ = ["GenRateTracker", "estimate_tokens_from_text", "CharsPerTokenEstimator"]


@dataclass
class GenRateTracker:
    """
    Accumulate tokens between UI samples.

    Call ``sample_interval()`` at a fixed cadence (e.g. once per second) to obtain
    tok/s = tokens in that period / elapsed wall time. Avoids inflated spikes when
    many stream chunks are handled in one event-loop tick (tiny dt).
    """

    _period_start: Optional[float] = field(default=None, init=False)
    _tokens_in_period: int = field(default=0, init=False)
    _last_rate: Optional[float] = field(default=None, init=False)

    def reset(self) -> None:
        self._period_start = None
        self._tokens_in_period = 0
        self._last_rate = None

    @property
    def rate(self) -> Optional[float]:
        return self._last_rate

    def add_tokens(self, n: int) -> Optional[float]:
        """Record tokens toward the current sampling period (does not update rate)."""
        if n <= 0:
            return self._last_rate
        if self._period_start is None:
            self._period_start = time.monotonic()
        self._tokens_in_period += n
        return self._last_rate

    def clear_period(self) -> None:
        """Drop in-flight period tokens without clearing the last sampled rate."""
        self._period_start = None
        self._tokens_in_period = 0

    def sample_interval(self, *, min_elapsed: float = 0.25) -> Optional[float]:
        """Close the current period and return tok/s (tokens / wall time)."""
        now = time.monotonic()
        if self._period_start is None or self._tokens_in_period <= 0:
            return self._last_rate
        elapsed = now - self._period_start
        if elapsed < min_elapsed:
            return self._last_rate
        self._last_rate = self._tokens_in_period / elapsed
        self._period_start = now
        self._tokens_in_period = 0
        return self._last_rate
