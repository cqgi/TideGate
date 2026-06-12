from __future__ import annotations

import random

from tidegate.config.models import DeploymentConfig, GatewayConfig, ModelGroupConfig
from tidegate.core.errors import ErrorCategory, GatewayError
from tidegate.routing.breaker import BreakerState
from tidegate.routing.stats import RoutingState, deployment_key


class NoAvailableDeployment(GatewayError):
    def __init__(self) -> None:
        super().__init__("no deployment available", ErrorCategory.RETRYABLE_UPSTREAM)


class P2CSelector:
    def __init__(self, settings: GatewayConfig, routing_state: RoutingState) -> None:
        self._settings = settings
        self._state = routing_state

    def pick(
        self,
        group: ModelGroupConfig,
        exclude: set[tuple[str, str]],
        *,
        now_s: float,
    ) -> DeploymentConfig:
        # SPEC-M3-3: candidate checks are side-effect free; dispatch records the
        # HALF_OPEN probe only after quota reservation succeeds.
        candidates = [
            deployment
            for deployment in group.deployments
            if self._candidate_allowed(deployment, exclude, now_s)
        ]
        if not candidates:
            raise NoAvailableDeployment()
        half_open = [
            deployment
            for deployment in candidates
            if self._state.refresh_breaker(deployment, now_s) == BreakerState.HALF_OPEN
        ]
        if half_open:
            return half_open[0]
        if len(candidates) == 1:
            return candidates[0]
        pair = random.sample(candidates, 2)
        scored = sorted(pair, key=lambda deployment: self._score(deployment, pair))
        return scored[0]

    def _candidate_allowed(
        self,
        deployment: DeploymentConfig,
        exclude: set[tuple[str, str]],
        now_s: float,
    ) -> bool:
        if deployment.weight <= 0 or deployment_key(deployment) in exclude:
            return False
        stats = self._state.stats_for(deployment)
        if stats.cooldown_until_s > now_s:
            return False
        self._state.refresh_breaker(deployment, now_s)
        return self._state.breaker_for(deployment).can_try(now_s)

    def _score(self, deployment: DeploymentConfig, candidates: list[DeploymentConfig]) -> float:
        weights = self._settings.routing.p2c_weights
        return (
            weights["ttft"] * self._norm(deployment, candidates, "ttft")
            + weights["error_rate"] * self._state.stats_for(deployment).ewma_error_rate
            + weights["inflight"] * self._norm(deployment, candidates, "inflight")
            + weights["price"] * self._norm(deployment, candidates, "price")
        )

    def _norm(
        self,
        deployment: DeploymentConfig,
        candidates: list[DeploymentConfig],
        field: str,
    ) -> float:
        values = [self._value(candidate, field) for candidate in candidates]
        value = self._value(deployment, field)
        low = min(values)
        high = max(values)
        if high == low:
            return 0.0
        return (value - low) / (high - low)

    def _value(self, deployment: DeploymentConfig, field: str) -> float:
        stats = self._state.stats_for(deployment)
        if field == "ttft":
            return stats.ewma_ttft_s
        if field == "inflight":
            return float(stats.inflight)
        if field == "price":
            return deployment.price_per_1k_input_usd + deployment.price_per_1k_output_usd
        raise ValueError(field)


def pick(group: ModelGroupConfig, exclude: set[tuple[str, str]]) -> DeploymentConfig:
    candidates = [
        deployment
        for deployment in group.deployments
        if (deployment.provider, deployment.upstream_model) not in exclude and deployment.weight > 0
    ]
    if not candidates:
        raise NoAvailableDeployment()
    total = sum(deployment.weight for deployment in candidates)
    target = random.uniform(0, total)
    cursor = 0.0
    for deployment in candidates:
        cursor += deployment.weight
        if cursor >= target:
            return deployment
    return candidates[-1]
