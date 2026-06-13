from __future__ import annotations

from pathlib import Path

import pytest
import redis.asyncio as redis
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.integration.test_quota import FixedEstimator
from tidegate.config.loader import load_config
from tidegate.config.models import GatewayConfig
from tidegate.core.models import ChatMessage, UnifiedRequest, Usage
from tidegate.quota.estimator import Estimate
from tidegate.quota.scripts import QuotaScripts
from tidegate.quota.service import QuotaReservation, QuotaService


@pytest.mark.integration
@pytest.mark.asyncio
@settings(max_examples=20, deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=8), min_size=50, max_size=120))
async def test_quota_conservation(redis_stack_proc: None, actual_tokens: list[int]) -> None:
    del redis_stack_proc
    snapshot = _snapshot()
    client = redis.Redis.from_url("redis://127.0.0.1:6379/0", decode_responses=False)
    await client.flushdb()
    scripts = QuotaScripts(client)
    await scripts.load()
    estimate = Estimate(prompt_tokens=1, output_tokens=9, tpm_cost=10, budget_cost_micro=1)
    quota = QuotaService(client, scripts, FixedEstimator(estimate))
    tenant = snapshot.tenants[0]
    deployment = snapshot.model_groups["chat-large"].deployments[0]
    initial_tpm = snapshot.quota_plans[tenant.plan].tpm
    accepted: list[tuple[QuotaReservation, int]] = []

    for index, actual in enumerate(actual_tokens):
        try:
            reservation = await quota.reserve(
                tenant=tenant,
                req=_req(f"prop-{index}"),
                deployment=deployment,
                snapshot=snapshot,
            )
        except Exception:
            continue
        accepted.append((reservation, actual))

    for reservation, actual in accepted:
        await quota.settle(
            reservation,
            Usage(prompt_tokens=1, completion_tokens=actual, total_tokens=actual + 1),
            actual,
        )

    state = await quota.debug_state(tenant.id)
    await client.aclose()
    expected_spent = sum(actual + 1 for _, actual in accepted)

    assert state["conc"] == 0
    assert state["resv"] == 0
    tpm_tokens = state["tpm_tokens"]
    assert isinstance(tpm_tokens, float)
    # The Lua bucket refills between reservations using real now_ms; conservation is
    # checked within that bounded refill error rather than by freezing server time.
    assert abs((initial_tpm - tpm_tokens) - expected_spent) < 2.0


def _snapshot() -> GatewayConfig:
    base = load_config(Path("tests/fixtures/gateway-test.yaml"))
    return base.model_copy(
        update={
            "quota_plans": {
                "free": base.quota_plans["free"].model_copy(
                    update={"rpm": 1000, "tpm": 1000, "concurrent_streams": 200}
                )
            }
        }
    )


def _req(request_id: str) -> UnifiedRequest:
    return UnifiedRequest(
        request_id=request_id,
        tenant_id="demo",
        model="chat-large",
        messages=[ChatMessage(role="user", content="hi")],
        stream=False,
        raw_body={"model": "chat-large", "messages": [{"role": "user", "content": "hi"}]},
    )
