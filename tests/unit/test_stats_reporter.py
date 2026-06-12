from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tidegate.config.loader import load_config
from tidegate.routing.breaker import BreakerState
from tidegate.routing.reporter import prewarm_from_aggregate, report_once
from tidegate.routing.stats import RoutingState


def test_slow_ttft_feeds_breaker_failure() -> None:
    """SPEC-M3-1."""
    settings = load_config(Path("tests/fixtures/gateway-test.yaml"))
    state = RoutingState(settings)
    deployment = settings.model_groups["chat-large"].deployments[0]

    for _ in range(settings.routing.breaker.min_samples):
        state.record_start(deployment, now_s=0.0)
        state.record_finish(deployment, success=True, ttft_s=1.0, now_s=0.0)

    assert state.breaker_for(deployment).state == BreakerState.OPEN


def test_rate_limit_cools_down_without_opening_breaker() -> None:
    """SPEC-M3-1."""
    settings = load_config(Path("tests/fixtures/gateway-test.yaml"))
    state = RoutingState(settings)
    deployment = settings.model_groups["chat-large"].deployments[0]

    state.record_start(deployment, now_s=0.0)
    state.record_rate_limit(deployment, retry_after_s=2.0, now_s=10.0)

    stats = state.stats_for(deployment)
    assert state.breaker_for(deployment).state == BreakerState.CLOSED
    assert stats.inflight == 0
    assert stats.cooldown_until_s == 12.0


def test_error_rate_ewma_bootstrap_distinguishes_initialized_zero() -> None:
    """REWORK-M3-1."""
    settings = load_config(Path("tests/fixtures/gateway-test.yaml"))
    state = RoutingState(settings)
    deployment = settings.model_groups["chat-large"].deployments[0]
    alpha = settings.routing.ewma_alpha

    state.record_start(deployment, now_s=0.0)
    state.record_finish(deployment, success=True, ttft_s=0.01, now_s=0.0)
    state.record_start(deployment, now_s=0.0)
    state.record_finish(deployment, success=False, ttft_s=0.01, now_s=0.0)

    assert state.stats_for(deployment).ewma_error_rate == pytest.approx(alpha)


@pytest.mark.asyncio
async def test_report_once_writes_aggregate_keys() -> None:
    """SPEC-M3-5."""
    settings = load_config(Path("tests/fixtures/gateway-test.yaml"))
    state = RoutingState(settings)
    deployment = settings.model_groups["chat-large"].deployments[0]
    state.record_start(deployment, now_s=0.0)
    state.record_finish(deployment, success=False, ttft_s=0.01, now_s=0.0)
    redis = FakeRedis()

    await report_once(redis, settings, state, instance="unit:1")  # type: ignore[arg-type]

    cb_key = "agg:cb:mock-a:mock-gpt-large"
    stats_key = "agg:stats:mock-a:mock-gpt-large"
    assert cb_key in redis.hashes
    assert stats_key in redis.hashes
    assert redis.ttls[cb_key] == settings.routing.agg_ttl_s
    assert redis.ttls[stats_key] == settings.routing.agg_ttl_s
    cb_value = json.loads(redis.hashes[cb_key]["unit:1"])
    assert cb_value["state"] == BreakerState.CLOSED.value


@pytest.mark.asyncio
async def test_prewarm_opens_when_majority_instances_open() -> None:
    """SPEC-M3-5."""
    settings = load_config(Path("tests/fixtures/gateway-test.yaml"))
    state = RoutingState(settings)
    deployment = settings.model_groups["chat-large"].deployments[0]
    redis = FakeRedis()
    redis.hashes["agg:cb:mock-a:mock-gpt-large"] = {
        "a": json.dumps({"state": "open", "remaining_s": 1.0}),
        "b": json.dumps({"state": "open", "remaining_s": 3.0}),
        "c": json.dumps({"state": "closed", "remaining_s": 0.0}),
    }

    await prewarm_from_aggregate(redis, settings, state, now_s=5.0)  # type: ignore[arg-type]

    breaker = state.breaker_for(deployment)
    assert breaker.state == BreakerState.OPEN
    assert breaker.open_until_s == 7.0


class FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.ttls: dict[str, int] = {}

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self)

    async def hvals(self, key: str) -> list[str]:
        return list(self.hashes.get(key, {}).values())


class FakePipeline:
    def __init__(self, redis: FakeRedis) -> None:
        self._redis = redis

    def hset(self, key: str, field: str, value: str) -> None:
        self._redis.hashes.setdefault(key, {})[field] = value

    def expire(self, key: str, ttl: int) -> None:
        self._redis.ttls[key] = ttl

    async def execute(self) -> list[Any]:
        return []
