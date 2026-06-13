from __future__ import annotations

import pytest

from tidegate.config.loader import load_config
from tidegate.core.errors import GatewayError
from tidegate.routing.selector import P2CSelector
from tidegate.routing.stats import RoutingState


def test_p2c_excludes_open_breaker() -> None:
    settings = load_config("tests/fixtures/gateway-test.yaml")
    state = RoutingState(settings)
    group = settings.model_groups["chat-large"]
    first, second = group.deployments
    state.breaker_for(first).record(success=False, now_s=0.0)
    state.breaker_for(first).record(success=True, now_s=0.0)
    state.breaker_for(first).record(success=True, now_s=0.0)
    state.breaker_for(first).record(success=True, now_s=0.0)
    state.breaker_for(first).record(success=True, now_s=0.0)
    state.breaker_for(first).record(success=False, now_s=0.0)
    state.breaker_for(first).record(success=False, now_s=0.0)
    state.breaker_for(first).record(success=False, now_s=0.0)
    state.breaker_for(first).record(success=False, now_s=0.0)
    state.breaker_for(first).record(success=False, now_s=0.0)

    picked = P2CSelector(settings, state).pick(group, set(), now_s=0.0)

    assert picked == second


def test_p2c_prefers_lower_score(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = load_config("tests/fixtures/gateway-test.yaml")
    state = RoutingState(settings)
    group = settings.model_groups["chat-large"]
    first, second = group.deployments
    state.stats_for(first).ewma_error_rate = 1.0
    state.stats_for(first).inflight = 10
    state.stats_for(second).ewma_error_rate = 0.0
    monkeypatch.setattr("tidegate.routing.selector.random.sample", lambda items, count: list(items))

    picked = P2CSelector(settings, state).pick(group, set(), now_s=0.0)

    assert picked == second


def test_p2c_empty_candidates_raises() -> None:
    settings = load_config("tests/fixtures/gateway-test.yaml")
    state = RoutingState(settings)
    group = settings.model_groups["chat-large"]
    exclude = {(deployment.provider, deployment.upstream_model) for deployment in group.deployments}

    with pytest.raises(GatewayError):
        P2CSelector(settings, state).pick(group, exclude, now_s=0.0)
