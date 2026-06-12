from __future__ import annotations

from pathlib import Path

import pytest
import redis.asyncio as redis

from tidegate.config.loader import load_config
from tidegate.core.models import Usage
from tidegate.obs.metrics import Metrics
from tidegate.quota.estimator import Estimate, QuotaEstimator
from tidegate.quota.service import QuotaReservation, QuotaService


class FakeScripts:
    async def settle(self, keys: list[str], args: list[str]) -> list[object]:
        del keys, args
        raise redis.RedisError("down")


class FakeEstimator(QuotaEstimator):
    async def update_correction(
        self,
        *,
        tenant_id: str,
        model: str,
        estimate: Estimate,
        actual: Usage,
        snapshot: object,
    ) -> None:
        del tenant_id, model, estimate, actual, snapshot


@pytest.mark.asyncio
async def test_settle_double_failure_is_swallowed_and_counted() -> None:
    """REWORK-M2-2."""
    snapshot = load_config(Path("tests/fixtures/gateway-test.yaml"))
    metrics = Metrics.create()
    quota = QuotaService(
        redis.Redis.from_url("redis://127.0.0.1:6379/0"),
        FakeScripts(),  # type: ignore[arg-type]
        FakeEstimator(None),  # type: ignore[arg-type]
        metrics=metrics,
    )
    reservation = QuotaReservation(
        tenant_id="demo",
        request_id="req",
        model="chat-large",
        deployment=snapshot.model_groups["chat-large"].deployments[0],
        estimate=Estimate(prompt_tokens=1, output_tokens=1, tpm_cost=2, budget_cost_micro=1),
        snapshot=snapshot,
        month="2026-06",
    )

    await quota.settle(reservation, Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2), 1)

    rendered = metrics.render()[0].decode()
    assert 'tidegate_quota_settle_failed_total{tenant="demo"} 1.0' in rendered
