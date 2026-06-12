from __future__ import annotations

from collections import deque
from enum import StrEnum

from tidegate.config.models import BreakerConfig


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, config: BreakerConfig) -> None:
        self._config = config
        self._state = BreakerState.CLOSED
        self._window: deque[bool] = deque(maxlen=config.window_size)
        self._open_count = 0
        self._open_until_s = 0.0
        self._half_open_inflight = 0
        self._half_open_successes = 0

    @property
    def state(self) -> BreakerState:
        return self._state

    @property
    def open_until_s(self) -> float:
        return self._open_until_s

    def can_try(self, now_s: float) -> bool:
        self.refresh(now_s)
        if self._state == BreakerState.CLOSED:
            return True
        return (
            self._state == BreakerState.HALF_OPEN
            and self._half_open_inflight < self._config.half_open_probes
        )

    def allow(self, now_s: float) -> bool:
        self.refresh(now_s)
        if self._state == BreakerState.CLOSED:
            return True
        if self._state == BreakerState.HALF_OPEN:
            if self._half_open_inflight >= self._config.half_open_probes:
                return False
            self._half_open_inflight += 1
            return True
        return False

    def refresh(self, now_s: float) -> BreakerState:
        if self._state == BreakerState.OPEN:
            if now_s < self._open_until_s:
                return self._state
            self._state = BreakerState.HALF_OPEN
            self._half_open_inflight = 0
            self._half_open_successes = 0
        return self._state

    def record(self, *, success: bool, now_s: float) -> BreakerState:
        if self._state == BreakerState.HALF_OPEN:
            self._half_open_inflight = max(0, self._half_open_inflight - 1)
            if success:
                self._half_open_successes += 1
                if self._half_open_successes >= self._config.half_open_probes:
                    self._close()
            else:
                self._open(now_s)
            return self._state

        self._window.append(success)
        if self._state == BreakerState.CLOSED and self._should_open():
            self._open(now_s)
        return self._state

    def snapshot(self) -> dict[str, object]:
        return {
            "state": self._state.value,
            "open_until_s": self._open_until_s,
            "open_count": self._open_count,
            "half_open_inflight": self._half_open_inflight,
        }

    def force_open(self, *, now_s: float, remaining_s: float) -> None:
        self._state = BreakerState.OPEN
        self._open_until_s = now_s + max(0.0, remaining_s)
        self._half_open_inflight = 0
        self._half_open_successes = 0

    def _should_open(self) -> bool:
        if len(self._window) < self._config.min_samples:
            return False
        failures = sum(1 for success in self._window if not success)
        return failures / len(self._window) >= self._config.failure_rate_to_open

    def _open(self, now_s: float) -> None:
        self._state = BreakerState.OPEN
        self._open_count += 1
        cooldown = min(
            self._config.cooldown_max_s,
            self._config.open_cooldown_s * (2 ** max(0, self._open_count - 1)),
        )
        self._open_until_s = now_s + cooldown
        self._half_open_inflight = 0
        self._half_open_successes = 0

    def _close(self) -> None:
        self._state = BreakerState.CLOSED
        self._window.clear()
        self._open_count = 0
        self._half_open_inflight = 0
        self._half_open_successes = 0
