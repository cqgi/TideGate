from __future__ import annotations

from dataclasses import dataclass

from tidegate.config.models import DeploymentConfig, GatewayConfig
from tidegate.core.errors import ErrorCategory, GatewayError
from tidegate.obs.metrics import Metrics
from tidegate.routing.breaker import BreakerState, CircuitBreaker


@dataclass
class DeploymentStats:
    ewma_ttft_s: float = 0.0
    ewma_error_rate: float = 0.0
    ttft_initialized: bool = False
    error_rate_initialized: bool = False
    inflight: int = 0
    cooldown_until_s: float = 0.0


class RoutingState:
    def __init__(self, settings: GatewayConfig, metrics: Metrics | None = None) -> None:
        self._settings = settings
        self._metrics = metrics
        self._stats: dict[tuple[str, str], DeploymentStats] = {}
        self._breakers: dict[tuple[str, str], CircuitBreaker] = {}
        for group in settings.model_groups.values():
            for deployment in group.deployments:
                key = deployment_key(deployment)
                self._stats[key] = DeploymentStats()
                self._breakers[key] = CircuitBreaker(settings.routing.breaker)
                self._set_breaker_metric(key, BreakerState.CLOSED)

    def stats_for(self, deployment: DeploymentConfig) -> DeploymentStats:
        key = deployment_key(deployment)
        return self._stats.setdefault(key, DeploymentStats())

    def breaker_for(self, deployment: DeploymentConfig) -> CircuitBreaker:
        key = deployment_key(deployment)
        breaker = self._breakers.get(key)
        if breaker is None:
            breaker = CircuitBreaker(self._settings.routing.breaker)
            self._breakers[key] = breaker
        return breaker

    def record_start(self, deployment: DeploymentConfig, *, now_s: float) -> None:
        if not self.breaker_for(deployment).allow(now_s):
            raise GatewayError("no deployment available", ErrorCategory.RETRYABLE_UPSTREAM)
        self.stats_for(deployment).inflight += 1

    def record_abort(self, deployment: DeploymentConfig) -> None:
        stats = self.stats_for(deployment)
        stats.inflight = max(0, stats.inflight - 1)

    def record_finish(
        self,
        deployment: DeploymentConfig,
        *,
        success: bool,
        ttft_s: float | None,
        now_s: float,
    ) -> None:
        stats = self.stats_for(deployment)
        stats.inflight = max(0, stats.inflight - 1)
        alpha = self._settings.routing.ewma_alpha
        if ttft_s is not None:
            stats.ewma_ttft_s = _ewma(
                stats.ewma_ttft_s,
                ttft_s,
                alpha,
                initialized=stats.ttft_initialized,
            )
            stats.ttft_initialized = True
            if ttft_s > self._settings.routing.slow_call_ttft_slo_s:
                success = False
        stats.ewma_error_rate = _ewma(
            stats.ewma_error_rate,
            0.0 if success else 1.0,
            alpha,
            initialized=stats.error_rate_initialized,
        )
        stats.error_rate_initialized = True
        breaker = self.breaker_for(deployment)
        before = breaker.state
        after = breaker.record(success=success, now_s=now_s)
        if after != before:
            self._record_transition(deployment_key(deployment), after)

    def record_rate_limit(
        self,
        deployment: DeploymentConfig,
        retry_after_s: float,
        now_s: float,
    ) -> None:
        self.stats_for(deployment).inflight = max(0, self.stats_for(deployment).inflight - 1)
        self.stats_for(deployment).cooldown_until_s = max(
            self.stats_for(deployment).cooldown_until_s,
            now_s + retry_after_s,
        )

    def refresh_breaker(self, deployment: DeploymentConfig, now_s: float) -> BreakerState:
        breaker = self.breaker_for(deployment)
        before = breaker.state
        after = breaker.refresh(now_s)
        if after != before:
            self._record_transition(deployment_key(deployment), after)
        return after

    def force_open(
        self,
        deployment: DeploymentConfig,
        *,
        now_s: float,
        remaining_s: float,
    ) -> None:
        breaker = self.breaker_for(deployment)
        before = breaker.state
        breaker.force_open(now_s=now_s, remaining_s=remaining_s)
        if breaker.state != before:
            self._record_transition(deployment_key(deployment), breaker.state)

    def snapshot(self) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for key, stats in self._stats.items():
            breaker = self._breakers[key]
            result[f"{key[0]}/{key[1]}"] = {
                "state": breaker.state.value,
                "open_until_s": breaker.open_until_s,
                "ewma_ttft": stats.ewma_ttft_s,
                "ewma_err": stats.ewma_error_rate,
                "inflight": stats.inflight,
                "cooldown_until_s": stats.cooldown_until_s,
            }
        return result

    def _record_transition(self, key: tuple[str, str], to_state: BreakerState) -> None:
        self._set_breaker_metric(key, to_state)
        if self._metrics is not None:
            self._metrics.breaker_transitions.labels(key[0], key[1], to_state.value).inc()

    def _set_breaker_metric(self, key: tuple[str, str], state: BreakerState) -> None:
        if self._metrics is None:
            return
        self._metrics.breaker_state.labels(key[0], key[1]).set(_state_value(state))


def deployment_key(deployment: DeploymentConfig) -> tuple[str, str]:
    return deployment.provider, deployment.upstream_model


def _ewma(current: float, sample: float, alpha: float, *, initialized: bool) -> float:
    if not initialized:
        return sample
    return alpha * sample + (1 - alpha) * current


def _state_value(state: BreakerState) -> int:
    match state:
        case BreakerState.CLOSED:
            return 0
        case BreakerState.HALF_OPEN:
            return 1
        case BreakerState.OPEN:
            return 2
