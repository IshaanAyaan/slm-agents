"""Sliding-window rate limiting."""

import time
from collections import deque

RATE_LIMIT_BURST = 20
RATE_LIMIT_WINDOW_SECONDS = 60.0


class RateLimiter:
    """Per-key sliding window limiter used by the API layer."""

    def __init__(self, burst: int = RATE_LIMIT_BURST) -> None:
        self._burst = burst
        self._events: dict[str, deque] = {}

    def allow(self, key: str) -> bool:
        """Return True when the call is within budget."""
        now = time.time()
        window = self._events.setdefault(key, deque())
        while window and now - window[0] > RATE_LIMIT_WINDOW_SECONDS:
            window.popleft()
        if len(window) >= self._burst:
            return False
        window.append(now)
        return True
