from __future__ import annotations

from pathlib import Path

import pytest
import redis.asyncio as redis

from tests.integration.test_quota import FixedEstimator
from tidegate.config.loader import load_config
from tidegate.core.models import ChatMessage, UnifiedRequest
from tidegate.quota.estimator import Estimate
from tidegate.quota.keys import reservation_zset_key
from tidegate.quota.scripts import QuotaScripts
from tidegate.quota.service import QuotaService


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sweep_refunds_expired_reservation(redis_stack_proc: None) -> None:
    """SPEC-M2-4."""
    del redis_stack_proc
    snapshot = load_config(Path("tests/fixtures/gateway-test.yaml")).model_copy(
        update={
            "sweeper": load_config(Path("tests/fixtures/gateway-test.yaml")).sweeper.model_copy(
                update={"batch_limit": 100}
            )
        }
    )
    client = redis.Redis.from_url("redis://127.0.0.1:6379/0", decode_responses=False)
    scripts = QuotaScripts(client)
    await scripts.load()
    estimate = Estimate(prompt_tokens=1, output_tokens=9, tpm_cost=10, budget_cost_micro=1)
    quota = QuotaService(client, scripts, FixedEstimator(estimate))
    tenant = snapshot.tenants[0]
    deployment = snapshot.model_groups["chat-large"].deployments[0]

    await quota.reserve(
        tenant=tenant,
        req=_req("sweep-req"),
        deployment=deployment,
        snapshot=snapshot,
    )
    await client.zadd(reservation_zset_key(tenant.id), {"sweep-req": 0})

    swept = await quota.sweep_tenant(snapshot, tenant)
    state = await quota.debug_state(tenant.id)
    await client.aclose()

    assert swept == 1
    assert state["conc"] == 0
    assert state["resv"] == 0


def _req(request_id: str) -> UnifiedRequest:
    return UnifiedRequest(
        request_id=request_id,
        tenant_id="demo",
        model="chat-large",
        messages=[ChatMessage(role="user", content="hi")],
        stream=False,
        raw_body={"model": "chat-large", "messages": [{"role": "user", "content": "hi"}]},
    )
