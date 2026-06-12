from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


@dataclass(frozen=True)
class Metrics:
    registry: CollectorRegistry
    requests: Counter
    ttft: Histogram
    overhead: Histogram
    upstream_aborted: Counter
    retry: Counter
    loop_lag: Gauge
    tokens: Counter
    cost: Counter
    quota_rejections: Counter
    quota_sweep: Counter
    quota_settle_failed: Counter
    breaker_state: Gauge
    breaker_transitions: Counter
    cache_events: Counter

    @classmethod
    def create(cls) -> Metrics:
        registry = CollectorRegistry()
        return cls(
            registry=registry,
            requests=Counter(
                "tidegate_requests",
                "Gateway requests",
                ("tenant", "model", "outcome"),
                registry=registry,
            ),
            ttft=Histogram(
                "tidegate_ttft_seconds",
                "Upstream time to first token",
                # REWORK-M0-1: match contract labels and avoid tenant-cardinality histograms.
                ("provider", "model"),
                registry=registry,
            ),
            overhead=Histogram(
                "tidegate_gateway_overhead_seconds",
                "Gateway overhead before dispatching upstream",
                # REWORK-M0-1: contract defines gateway overhead as unlabeled pre-upstream SLI.
                (),
                registry=registry,
            ),
            upstream_aborted=Counter(
                "tidegate_upstream_aborted",
                "Upstream streams aborted by the gateway",
                # REWORK-M0-1: contract labels are provider and reason.
                ("provider", "reason"),
                registry=registry,
            ),
            retry=Counter(
                "tidegate_retry",
                "Gateway retry attempts",
                ("reason",),
                registry=registry,
            ),
            loop_lag=Gauge(
                "tidegate_loop_lag_seconds",
                "Event loop scheduling lag",
                registry=registry,
            ),
            tokens=Counter(
                "tidegate_tokens",
                "Settled token usage",
                ("tenant", "model", "direction"),
                registry=registry,
            ),
            cost=Counter(
                "tidegate_cost_microusd",
                "Settled cost in micro-USD",
                ("tenant", "model"),
                registry=registry,
            ),
            quota_rejections=Counter(
                "tidegate_quota_rejections",
                "Quota rejections",
                ("tenant", "dim"),
                registry=registry,
            ),
            quota_sweep=Counter(
                "tidegate_quota_sweep",
                "Swept quota reservations",
                ("tenant",),
                registry=registry,
            ),
            quota_settle_failed=Counter(
                "tidegate_quota_settle_failed",
                "Quota settlement failures left for sweeper recovery",
                ("tenant",),
                registry=registry,
            ),
            breaker_state=Gauge(
                "tidegate_breaker_state",
                "Circuit breaker state by deployment: 0=closed, 1=half_open, 2=open",
                ("provider", "model"),
                registry=registry,
            ),
            breaker_transitions=Counter(
                "tidegate_breaker_transitions",
                "Circuit breaker transitions",
                ("provider", "model", "to_state"),
                registry=registry,
            ),
            cache_events=Counter(
                "tidegate_cache_events",
                "Cache events",
                ("level", "event"),
                registry=registry,
            ),
        )

    def render(self) -> tuple[bytes, str]:
        # REWORK-M0-2: generate_latest emits classic Prometheus text format.
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
