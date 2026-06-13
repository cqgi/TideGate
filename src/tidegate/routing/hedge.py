from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from tidegate.config.models import DeploymentConfig, HedgingConfig
from tidegate.routing.stats import RoutingState


@dataclass
class HedgeBudget:
    window_s: float = 60.0
    _requests: deque[float] = field(default_factory=deque)
    _hedges: deque[float] = field(default_factory=deque)

    def record_request(self, now_s: float | None = None) -> None:
        now = time.monotonic() if now_s is None else now_s
        self._prune(now)
        self._requests.append(now)

    def allow(self, config: HedgingConfig, now_s: float | None = None) -> bool:
        now = time.monotonic() if now_s is None else now_s
        self._prune(now)
        requests = max(1, len(self._requests))
        if (len(self._hedges) + 1) / requests > config.max_hedge_ratio:
            return False
        self._hedges.append(now)
        return True

    def _prune(self, now_s: float) -> None:
        cutoff = now_s - self.window_s
        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()
        while self._hedges and self._hedges[0] < cutoff:
            self._hedges.popleft()


def trigger_delay_s(
    deployment: DeploymentConfig,
    state: RoutingState,
    config: HedgingConfig,
) -> float:
    # Approximate local P95 with EWMA TTFT x 1.5 so hedging can reuse instance-local
    # stats instead of introducing a second latency statistics system.
    ewma = state.stats_for(deployment).ewma_ttft_s
    return max(config.trigger_floor_s, ewma * 1.5)
