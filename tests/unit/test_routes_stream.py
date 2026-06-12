from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import Request

import tidegate.api.routes as routes
from tidegate.config.loader import load_config
from tidegate.config.models import DeploymentConfig, GatewayConfig
from tidegate.core.deadline import Deadline
from tidegate.core.errors import ErrorCategory, GatewayError
from tidegate.core.models import (
    ChatCompletionIn,
    UnifiedDelta,
    UnifiedRequest,
    UnifiedResponse,
    Usage,
)
from tidegate.obs.metrics import Metrics
from tidegate.quota.estimator import Estimate
from tidegate.quota.service import QuotaReservation
from tidegate.routing.ladder import RoutingLadder
from tidegate.routing.selector import P2CSelector
from tidegate.routing.stats import RoutingState


def test_stale_cache_result_is_degraded_outcome() -> None:
    """SPEC-M4-7."""
    result = routes._ChatResult(_response(), "cache", "hit-semantic", "stale-cache")

    assert routes._outcome_for(result) == "degraded"


@pytest.mark.asyncio
async def test_stream_heartbeat_does_not_block_ttft_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """REWORK-M1-1 and REWORK-M1-4."""
    settings = _settings_for_stream_tests()
    request = _fake_request(
        settings,
        {
            "mock-a": _TtftTimeoutProvider(),
            "mock-b": _SuccessfulStreamProvider(),
        },
    )
    monkeypatch.setattr("tidegate.routing.selector.random.sample", lambda items, count: list(items))

    chunks = [
        chunk
        async for chunk in routes._stream_with_retries(
            request=cast(Request, request),
            unified=_unified_request(stream=True),
            deadline=_deadline(),
            incoming=_incoming(stream=True),
            levels=RoutingLadder(settings).levels("chat-large", settings.tenants[0]),
        )
    ]

    rendered_metrics = request.app.state.metrics.render()[0].decode()
    assert chunks[0] == b": ping\n\n"
    assert any(b"tok0" in chunk for chunk in chunks)
    assert chunks[-1] == b"data: [DONE]\n\n"
    assert (
        'tidegate_requests_total{model="chat-large",outcome="ok",tenant="demo"} 1.0'
        in rendered_metrics
    )
    assert 'tidegate_retry_total{reason="timeout_ttft"} 1.0' in rendered_metrics


@pytest.mark.asyncio
async def test_stream_retry_exhaustion_returns_error_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REWORK-M1-2."""
    settings = _settings_for_stream_tests()
    request = _fake_request(
        settings,
        {
            "mock-a": _ImmediateFailureProvider(),
            "mock-b": _ImmediateFailureProvider(),
        },
    )
    monkeypatch.setattr("tidegate.routing.selector.random.sample", lambda items, count: list(items))

    chunks = [
        chunk
        async for chunk in routes._stream_with_retries(
            request=cast(Request, request),
            unified=_unified_request(stream=True),
            deadline=_deadline(),
            incoming=_incoming(stream=True),
            levels=RoutingLadder(settings).levels("chat-large", settings.tenants[0]),
        )
    ]

    rendered_metrics = request.app.state.metrics.render()[0].decode()
    assert any(b'"finish_reason":"error"' in chunk for chunk in chunks)
    assert chunks[-1] == b"data: [DONE]\n\n"
    assert (
        'tidegate_requests_total{model="chat-large",outcome="error",tenant="demo"} 1.0'
        in rendered_metrics
    )


class _FakeRequest:
    def __init__(self, settings: GatewayConfig, providers: dict[str, object]) -> None:
        quota = _FakeQuotaService(settings)
        metrics = Metrics.create()
        routing_state = RoutingState(settings, metrics)
        self.app = SimpleNamespace(
            state=SimpleNamespace(
                config_holder=SimpleNamespace(current=settings),
                metrics=metrics,
                provider_manager=SimpleNamespace(providers=providers),
                quota=quota,
                cache=SimpleNamespace(lookup=_cache_miss, store=_cache_store_noop),
                routing_state=routing_state,
                selector=P2CSelector(settings, routing_state),
            )
        )
        self.state = SimpleNamespace(tenant=settings.tenants[0], request_id="req-test")

    async def is_disconnected(self) -> bool:
        return False


async def _cache_miss(*args: object, **kwargs: object) -> None:
    del args, kwargs


async def _cache_store_noop(*args: object, **kwargs: object) -> None:
    del args, kwargs


class _TtftTimeoutProvider:
    async def stream_chat(
        self,
        req: UnifiedRequest,
        upstream_model: str,
        deadline: Deadline,
    ) -> AsyncIterator[UnifiedDelta]:
        del req, upstream_model, deadline
        await asyncio.sleep(0.02)
        raise GatewayError("slow first token", ErrorCategory.TIMEOUT_TTFT)
        yield UnifiedDelta()

    async def chat(
        self,
        req: UnifiedRequest,
        upstream_model: str,
        deadline: Deadline,
    ) -> Any:
        del req, upstream_model, deadline
        raise AssertionError("not used")

    async def aclose(self) -> None:
        return None


class _ImmediateFailureProvider(_TtftTimeoutProvider):
    async def stream_chat(
        self,
        req: UnifiedRequest,
        upstream_model: str,
        deadline: Deadline,
    ) -> AsyncIterator[UnifiedDelta]:
        del req, upstream_model, deadline
        raise GatewayError("connection failed", ErrorCategory.RETRYABLE_UPSTREAM)
        yield UnifiedDelta()


class _SuccessfulStreamProvider(_TtftTimeoutProvider):
    async def stream_chat(
        self,
        req: UnifiedRequest,
        upstream_model: str,
        deadline: Deadline,
    ) -> AsyncIterator[UnifiedDelta]:
        del req, upstream_model, deadline
        yield UnifiedDelta(content="tok0 ")
        yield UnifiedDelta(finish_reason="stop")
        yield UnifiedDelta(usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2))


class _FakeQuotaService:
    def __init__(self, settings: GatewayConfig) -> None:
        self._settings = settings
        self.settled = 0

    async def reserve(self, **kwargs: object) -> QuotaReservation:
        req = kwargs["req"]
        deployment = kwargs["deployment"]
        assert isinstance(req, UnifiedRequest)
        assert isinstance(deployment, DeploymentConfig)
        return QuotaReservation(
            tenant_id=req.tenant_id,
            request_id=req.request_id,
            model=req.model,
            deployment=deployment,
            estimate=Estimate(prompt_tokens=1, output_tokens=1, tpm_cost=2, budget_cost_micro=1),
            snapshot=self._settings,
            month="2026-06",
        )

    async def settle(
        self,
        reservation: QuotaReservation,
        actual: Usage | None,
        forwarded_tokens: int,
    ) -> None:
        del reservation, actual, forwarded_tokens
        self.settled += 1


def _fake_request(settings: GatewayConfig, providers: dict[str, object]) -> _FakeRequest:
    return _FakeRequest(settings, providers)


def _response() -> UnifiedResponse:
    return UnifiedResponse(
        content="ok",
        finish_reason="stop",
        model="mock",
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _settings_for_stream_tests() -> GatewayConfig:
    settings = load_config(Path("tests/fixtures/gateway-test.yaml"))
    deployments = list(settings.model_groups["chat-large"].deployments)
    deployments[0] = deployments[0].model_copy(update={"weight": 100})
    return settings.model_copy(
        update={
            "server": settings.server.model_copy(update={"sse_heartbeat_interval_s": 0.01}),
            "routing": settings.routing.model_copy(
                update={
                    "max_attempts_before_first_byte": 2,
                    "p2c_weights": {
                        "ttft": 0.0,
                        "error_rate": 0.0,
                        "inflight": 0.0,
                        "price": 0.0,
                    },
                }
            ),
            "model_groups": {
                "chat-large": settings.model_groups["chat-large"].model_copy(
                    update={"deployments": tuple(deployments)}
                )
            },
        }
    )


def _incoming(*, stream: bool) -> ChatCompletionIn:
    return ChatCompletionIn.model_validate(
        {
            "model": "chat-large",
            "stream": stream,
            "messages": [{"role": "user", "content": "hi"}],
        }
    )


def _unified_request(*, stream: bool) -> UnifiedRequest:
    return UnifiedRequest(
        request_id="req-test",
        tenant_id="demo",
        model="chat-large",
        messages=_incoming(stream=stream).messages,
        stream=stream,
        raw_body={
            "model": "chat-large",
            "stream": stream,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )


def _deadline() -> Deadline:
    loop = asyncio.get_running_loop()
    return Deadline(
        connect_s=1.0,
        ttft_s=0.01,
        inter_chunk_s=1.0,
        total_deadline=loop.time() + 1.0,
    )
