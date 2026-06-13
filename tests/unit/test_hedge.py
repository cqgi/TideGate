from __future__ import annotations

import pytest

from tidegate.config.models import (
    DeploymentConfig,
    GatewayConfig,
    HedgingConfig,
    ModelGroupConfig,
    ProviderConfig,
    TenantConfig,
)
from tidegate.routing.hedge import HedgeBudget, trigger_delay_s
from tidegate.routing.stats import RoutingState


def test_hedge_budget_caps_ratio() -> None:
    budget = HedgeBudget(window_s=60)
    config = HedgingConfig(max_hedge_ratio=0.5)
    for index in range(10):
        budget.record_request(now_s=float(index))

    assert budget.allow(config, now_s=10.0)
    assert budget.allow(config, now_s=11.0)
    assert budget.allow(config, now_s=12.0)
    assert budget.allow(config, now_s=13.0)
    assert budget.allow(config, now_s=14.0)
    assert not budget.allow(config, now_s=16.0)


def test_trigger_delay_reuses_local_ewma_ttft() -> None:
    deployment = DeploymentConfig(provider="mock-a", upstream_model="mock")
    state = RoutingState(_settings())
    state.stats_for(deployment).ewma_ttft_s = 0.4

    assert trigger_delay_s(deployment, state, HedgingConfig(trigger_floor_s=0.1)) == pytest.approx(
        0.6
    )
    assert trigger_delay_s(deployment, state, HedgingConfig(trigger_floor_s=1.0)) == 1.0


def _settings() -> GatewayConfig:
    return GatewayConfig(
        providers={
            "mock-a": ProviderConfig(type="openai_compatible", base_url="", api_key_env="X")
        },
        model_groups={
            "chat-large": ModelGroupConfig(
                deployments=(DeploymentConfig(provider="mock-a", upstream_model="mock"),)
            )
        },
        tenants=(TenantConfig(id="demo", api_key_sha256="hash"),),
    )
