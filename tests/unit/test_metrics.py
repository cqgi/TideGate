from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST

from tidegate.obs.metrics import Metrics


def test_metrics_contract_names_and_labels() -> None:
    """REWORK-M0-1."""
    metrics = Metrics.create()
    metrics.requests.labels("demo", "chat-large", "client_disconnect").inc()
    metrics.ttft.labels("mock-a", "chat-large").observe(0.1)
    metrics.overhead.observe(0.01)
    metrics.upstream_aborted.labels("mock-a", "client_disconnect").inc()
    body, content_type = metrics.render()
    rendered = body.decode()
    assert content_type == CONTENT_TYPE_LATEST
    assert (
        'tidegate_requests_total{model="chat-large",outcome="client_disconnect",tenant="demo"}'
        in rendered
    )
    assert 'tidegate_ttft_seconds_count{model="chat-large",provider="mock-a"}' in rendered
    assert "tidegate_gateway_overhead_seconds_count" in rendered
    assert (
        'tidegate_upstream_aborted_total{provider="mock-a",reason="client_disconnect"}' in rendered
    )
