from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from tidegate.config.loader import load_config
from tidegate.core.models import ChatMessage, UnifiedRequest, Usage
from tidegate.quota.estimator import QuotaEstimator


class FakeCorrectionStore:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    async def hget(self, key: str, field: str) -> object | None:
        return self.values.get((key, field))

    async def hset(self, key: str, mapping: Mapping[str, str]) -> object:
        for field, value in mapping.items():
            self.values[(key, field)] = value
        return 1


@pytest.mark.asyncio
async def test_estimator_uses_prompt_max_tokens_and_price() -> None:
    """SPEC-M2-2."""
    snapshot = load_config(Path("tests/fixtures/gateway-test.yaml"))
    deployment = snapshot.model_groups["chat-large"].deployments[0]
    store = FakeCorrectionStore()
    estimator = QuotaEstimator(store)
    req = _request(max_tokens=5)

    estimate = await estimator.estimate(req, deployment, snapshot)

    assert estimate.prompt_tokens >= 1
    assert estimate.output_tokens == 5
    assert estimate.tpm_cost == estimate.prompt_tokens + estimate.output_tokens
    assert estimate.budget_cost_micro > 0


@pytest.mark.asyncio
async def test_estimator_updates_ratio_and_output_ewma() -> None:
    """SPEC-M2-2."""
    snapshot = load_config(Path("tests/fixtures/gateway-test.yaml"))
    deployment = snapshot.model_groups["chat-large"].deployments[0]
    store = FakeCorrectionStore()
    estimator = QuotaEstimator(store)
    req = _request(max_tokens=10)
    estimate = await estimator.estimate(req, deployment, snapshot)

    await estimator.update_correction(
        tenant_id="demo",
        model="chat-large",
        estimate=estimate,
        actual=Usage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        snapshot=snapshot,
    )

    key = "corr:demo:chat-large"
    assert float(store.values[(key, "ewma_ratio")]) > 0
    assert float(store.values[(key, "output_ewma")]) > 0


def _request(max_tokens: int | None) -> UnifiedRequest:
    return UnifiedRequest(
        request_id="req",
        tenant_id="demo",
        model="chat-large",
        messages=[ChatMessage(role="user", content="hello world")],
        stream=False,
        max_tokens=max_tokens,
        raw_body={"model": "chat-large", "messages": [{"role": "user", "content": "hello world"}]},
    )
