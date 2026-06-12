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
    loop_lag: Gauge

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
            loop_lag=Gauge(
                "tidegate_loop_lag_seconds",
                "Event loop scheduling lag",
                registry=registry,
            ),
        )

    def render(self) -> tuple[bytes, str]:
        # REWORK-M0-2: generate_latest emits classic Prometheus text format.
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
