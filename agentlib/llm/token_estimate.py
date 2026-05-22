"""
Heuristic token counts when the API does not stream live counts.

Starts at ~4.5 chars/token; refines from Ollama ``eval_count`` vs streamed text each
completion (in-memory only — not written to agent.json).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Default until Ollama reports eval_count on a completion.
DEFAULT_CHARS_PER_TOKEN = 4.5
MIN_CHARS_PER_TOKEN = 4.0
MAX_CHARS_PER_TOKEN = 12.0
# Smoothing so one odd turn (tool JSON, empty answer) does not swing estimates wildly.
CALIBRATION_EMA_ALPHA = 0.25
MIN_EVAL_COUNT_FOR_CALIBRATION = 8
MIN_CHARS_FOR_CALIBRATION = 16


@dataclass
class CharsPerTokenEstimator:
    """
    Session-local chars→tokens divisor learned from Ollama completions.

    Used for context-window heuristics, pre-usage streaming tok/s, and hosted backends.
    Live Ollama tok/s still prefers ``eval_count`` deltas once usage chunks arrive.
    """

    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN
    calibration_observations: int = 0
    last_observed_chars_per_token: Optional[float] = field(default=None)

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        cpt = max(1.0, float(self.chars_per_token))
        return max(1, (len(text) + int(cpt) - 1) // int(cpt))

    def char_cap_for_token_budget(self, max_tokens: int) -> int:
        """Upper bound on UTF-8 length for ``max_tokens`` heuristic tokens."""
        return max(200, int(max_tokens * max(1.0, float(self.chars_per_token))))

    def observe_ollama_completion(
        self,
        *,
        content: str = "",
        thinking: str = "",
        eval_count: int,
    ) -> None:
        """
        Update divisor from one finished Ollama generation.

        ``eval_count`` is Ollama's output token count (often includes thinking tokens).
        Char volume includes both ``content`` and ``thinking`` strings from the stream.
        """
        if not isinstance(eval_count, int) or eval_count < MIN_EVAL_COUNT_FOR_CALIBRATION:
            return
        total_chars = len(content or "") + len(thinking or "")
        if total_chars < MIN_CHARS_FOR_CALIBRATION:
            return
        observed = total_chars / float(eval_count)
        self.last_observed_chars_per_token = observed
        if self.calibration_observations == 0:
            blended = observed
        else:
            a = CALIBRATION_EMA_ALPHA
            blended = (1.0 - a) * self.chars_per_token + a * observed
        self.chars_per_token = max(MIN_CHARS_PER_TOKEN, min(MAX_CHARS_PER_TOKEN, blended))
        self.calibration_observations += 1

    @staticmethod
    def observe_from_assistant_message(
        estimator: "CharsPerTokenEstimator",
        msg: dict,
        usage: Optional[dict],
    ) -> None:
        if not usage or not isinstance(msg, dict):
            return
        ec = usage.get("eval_count")
        if not isinstance(ec, int):
            return
        content = msg.get("content")
        thinking = msg.get("thinking")
        estimator.observe_ollama_completion(
            content=content if isinstance(content, str) else "",
            thinking=thinking if isinstance(thinking, str) else "",
            eval_count=ec,
        )

    def calibration_note_for_usage(self) -> str:
        cpt = self.chars_per_token
        if self.calibration_observations <= 0:
            return (
                f"Token estimate heuristic: ~{cpt:.1f} chars/token (default {DEFAULT_CHARS_PER_TOKEN:g}; "
                "not calibrated yet — needs an Ollama completion with eval_count)."
            )
        last = self.last_observed_chars_per_token
        last_bit = f"; last completion ~{last:.1f} chars/token" if last is not None else ""
        n = self.calibration_observations
        return (
            f"Token estimate heuristic: ~{cpt:.1f} chars/token "
            f"(calibrated from {n} Ollama completion{'s' if n != 1 else ''}{last_bit}). "
            "Used for context sizing and estimates before live eval_count is available."
        )


_DEFAULT_ESTIMATOR = CharsPerTokenEstimator()


def get_default_chars_per_token_estimator() -> CharsPerTokenEstimator:
    return _DEFAULT_ESTIMATOR


def estimate_tokens_from_text(
    text: str,
    *,
    estimator: Optional[CharsPerTokenEstimator] = None,
) -> int:
    return (estimator or _DEFAULT_ESTIMATOR).estimate_tokens(text)
