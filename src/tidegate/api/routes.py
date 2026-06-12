from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from tidegate.api.sse import (
    DisconnectAwareStreamingResponse,
    StreamAccounting,
    error_chunk,
    heartbeat_event,
    log_stream_access,
    sse_event,
    stream_chunk_payload,
    with_heartbeats,
)
from tidegate.config.models import DeploymentConfig, GatewayConfig, ModelGroupConfig, TenantConfig
from tidegate.core.deadline import Deadline
from tidegate.core.errors import ErrorCategory, GatewayError
from tidegate.core.models import (
    ChatCompletionIn,
    UnifiedDelta,
    UnifiedRequest,
    UnifiedResponse,
    Usage,
)
from tidegate.providers.base import Provider
from tidegate.quota.service import QuotaReservation, QuotaService
from tidegate.routing.selector import pick

router = APIRouter()


class _ClientDisconnectedError(Exception):
    pass


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    body, content_type = request.app.state.metrics.render()
    return Response(body, media_type=content_type)


@router.get("/v1/models")
async def list_models(request: Request) -> JSONResponse:
    settings: GatewayConfig = request.app.state.config_holder.current
    models = [
        {"id": name, "object": "model", "created": 0, "owned_by": "tidegate"}
        for name in sorted(settings.model_groups)
    ]
    return JSONResponse({"object": "list", "data": models})


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> Response:
    settings: GatewayConfig = request.app.state.config_holder.current
    request_started = time.monotonic()
    try:
        raw_body = await request.json()
    except json.JSONDecodeError as exc:
        # REWORK-M0-4: malformed JSON is a client 422, not a server 500.
        raise GatewayError(
            "request body must be valid JSON",
            ErrorCategory.CLIENT_ERROR,
            http_status=422,
        ) from exc
    try:
        incoming = ChatCompletionIn.model_validate(raw_body)
    except ValidationError as exc:
        raise GatewayError(
            "request validation failed",
            ErrorCategory.CLIENT_ERROR,
            http_status=422,
        ) from exc

    group = settings.model_groups.get(incoming.model)
    if group is None:
        raise GatewayError("unknown model", ErrorCategory.CLIENT_ERROR, http_status=404)
    tenant: TenantConfig = request.state.tenant
    request_id: str = request.state.request_id
    deadline = _deadline(settings)
    unified = _unified_request(request, incoming, raw_body, tenant, request_id)
    # REWORK-M0-1: gateway overhead is request receipt through upstream dispatch.
    request.app.state.metrics.overhead.observe(time.monotonic() - request_started)
    if incoming.stream:
        stream_exclude: set[tuple[str, str]] = set()
        deployment, provider = _pick_attempt(request, group, stream_exclude)
        stream_exclude.add((deployment.provider, deployment.upstream_model))
        quota_settle = await _reserve_quota(
            request=request,
            deployment=deployment,
            unified=unified,
            snapshot=settings,
        )
        return DisconnectAwareStreamingResponse(
            _stream_with_retries(
                request=request,
                group=group,
                unified=unified,
                deadline=deadline,
                incoming=incoming,
                first_attempt=(deployment, provider, quota_settle),
                exclude=stream_exclude,
            ),
            media_type="text/event-stream",
            headers={
                "X-TideGate-Cache": "miss",
            },
        )

    started = time.monotonic()
    response, route_header = await _call_non_stream_with_retries(
        request=request,
        group=group,
        unified=unified,
        deadline=deadline,
        model=incoming.model,
    )
    duration = time.monotonic() - started
    request.app.state.metrics.requests.labels(tenant.id, incoming.model, "ok").inc()
    structlog.get_logger().info(
        "access",
        tenant=tenant.id,
        model=incoming.model,
        outcome="ok",
        route=route_header,
        ttft_ms=None,
        duration_ms=duration * 1000,
        forwarded_chars=len(response.content),
        forwarded_deltas=1,
        usage=response.usage.model_dump(),
    )
    return JSONResponse(
        _non_stream_payload(request_id, incoming.model, response),
        headers={"X-TideGate-Cache": "miss", "X-TideGate-Route": route_header},
    )


async def _render_stream(
    *,
    request: Request,
    upstream: AsyncIterator[UnifiedDelta],
    incoming: ChatCompletionIn,
    provider_name: str,
    accounting: StreamAccounting,
    quota_settle: _QuotaSettlement,
) -> AsyncIterator[bytes]:
    settings: GatewayConfig = request.app.state.config_holder.current
    request_id: str = request.state.request_id
    try:
        async for delta in with_heartbeats(upstream, settings.server.sse_heartbeat_interval_s):
            if await request.is_disconnected():
                # SPEC-M0-5: client disconnects close upstream streaming immediately.
                request.app.state.metrics.upstream_aborted.labels(
                    provider=provider_name, reason="client_disconnect"
                ).inc()
                raise _ClientDisconnectedError
            if delta is None:
                yield heartbeat_event()
                continue
            now = time.monotonic()
            if accounting.ttft_ms is None and delta.content:
                accounting.ttft_ms = (now - accounting.started_at) * 1000
                request.app.state.metrics.ttft.labels(provider_name, incoming.model).observe(
                    accounting.ttft_ms / 1000
                )
            _accumulate(accounting, delta)
            payload = stream_chunk_payload(
                request_id,
                incoming.model,
                delta,
                include_usage=incoming.include_stream_usage(),
            )
            if payload is not None:
                yield sse_event(payload)
        quota_settle.capture_usage(accounting.usage)
        yield sse_event("[DONE]")
    except asyncio.CancelledError:
        request.app.state.metrics.upstream_aborted.labels(
            provider=provider_name, reason="client_disconnect"
        ).inc()
        raise


async def _stream_with_retries(
    *,
    request: Request,
    group: ModelGroupConfig,
    unified: UnifiedRequest,
    deadline: Deadline,
    incoming: ChatCompletionIn,
    first_attempt: tuple[DeploymentConfig, Provider, _QuotaSettlement] | None = None,
    exclude: set[tuple[str, str]] | None = None,
) -> AsyncIterator[bytes]:
    settings: GatewayConfig = request.app.state.config_holder.current
    tenant: TenantConfig = request.state.tenant
    attempted = set() if exclude is None else set(exclude)
    attempts = settings.routing.max_attempts_before_first_byte
    last_error: GatewayError | None = None
    route_header = "none"
    accounting = StreamAccounting(started_at=time.monotonic())
    outcome = "error"
    attempt_count = 0
    quota_settle: _QuotaSettlement | None = None
    try:
        for attempt_index in range(attempts):
            attempt_count = attempt_index + 1
            if attempt_index == 0 and first_attempt is not None:
                deployment, provider, quota_settle = first_attempt
            else:
                try:
                    deployment, provider = _pick_attempt(request, group, attempted)
                    quota_settle = await _reserve_quota(
                        request=request,
                        deployment=deployment,
                        unified=unified,
                        snapshot=settings,
                        count_request_rejection=False,
                    )
                except GatewayError as exc:
                    last_error = exc
                    break
                attempted.add((deployment.provider, deployment.upstream_model))
            route_header = f"{deployment.provider}/{deployment.upstream_model}"
            upstream = provider.stream_chat(unified, deployment.upstream_model, deadline)
            sent_data = False
            try:
                async for chunk in _render_stream(
                    request=request,
                    upstream=upstream,
                    incoming=incoming,
                    provider_name=deployment.provider,
                    accounting=accounting,
                    quota_settle=quota_settle,
                ):
                    # DECISION: REWORK-M1-1 treats only SSE data events as the idempotency
                    # boundary; heartbeat comments can be followed by a safe TTFT retry.
                    if chunk.startswith(b"data:"):
                        sent_data = True
                    yield chunk
                outcome = "ok"
                return
            except _ClientDisconnectedError:
                outcome = "client_disconnect"
                return
            except GatewayError as exc:
                last_error = exc
                retryable = exc.category in {
                    ErrorCategory.RETRYABLE_UPSTREAM,
                    ErrorCategory.RATE_LIMITED_UPSTREAM,
                    ErrorCategory.TIMEOUT_TTFT,
                }
                if sent_data or not retryable or attempt_index == attempts - 1:
                    break
                request.app.state.metrics.retry.labels(reason=exc.category.value).inc()
                request.app.state.metrics.upstream_aborted.labels(
                    provider=deployment.provider, reason="timeout"
                ).inc()
                aclose = getattr(upstream, "aclose", None)
                if callable(aclose):
                    await aclose()
                await quota_settle.settle_once(None, accounting.delta_count)
                quota_settle = None
                continue
        if last_error is None:
            last_error = GatewayError("no deployment available", ErrorCategory.RETRYABLE_UPSTREAM)
        # REWORK-M1-2: once StreamingResponse has been selected, every failure is in-band.
        structlog.get_logger().error(
            "stream_request_failed",
            category=last_error.category.value,
            attempts=attempt_count,
            route=route_header,
        )
        async for chunk in error_chunk(unified.request_id, incoming.model):
            yield chunk
    except asyncio.CancelledError:
        outcome = "client_disconnect"
        raise
    finally:
        duration = time.monotonic() - accounting.started_at
        request.app.state.metrics.requests.labels(tenant.id, incoming.model, outcome).inc()
        log_stream_access(
            tenant=tenant.id,
            model=incoming.model,
            outcome=outcome,
            route=route_header,
            accounting=accounting,
            duration_ms=duration * 1000,
        )
        if quota_settle is not None:
            await quota_settle.settle_once(accounting.usage, accounting.delta_count)


async def _call_non_stream_with_retries(
    *,
    request: Request,
    group: ModelGroupConfig,
    unified: UnifiedRequest,
    deadline: Deadline,
    model: str,
) -> tuple[UnifiedResponse, str]:
    settings: GatewayConfig = request.app.state.config_holder.current
    exclude: set[tuple[str, str]] = set()
    last_error: GatewayError | None = None
    for _ in range(settings.routing.max_attempts_before_first_byte):
        deployment, provider = _pick_attempt(request, group, exclude)
        exclude.add((deployment.provider, deployment.upstream_model))
        route_header = f"{deployment.provider}/{deployment.upstream_model}"
        quota_settle = await _reserve_quota(
            request=request,
            deployment=deployment,
            unified=unified,
            snapshot=settings,
        )
        try:
            response = await provider.chat(unified, deployment.upstream_model, deadline)
            await quota_settle.settle_once(response.usage, response.usage.completion_tokens)
            return response, route_header
        except GatewayError as exc:
            await quota_settle.settle_once(None, 0)
            last_error = exc
            if exc.category not in {
                ErrorCategory.RETRYABLE_UPSTREAM,
                ErrorCategory.RATE_LIMITED_UPSTREAM,
                ErrorCategory.TIMEOUT_TTFT,
            }:
                request.app.state.metrics.requests.labels(unified.tenant_id, model, "error").inc()
                raise
            request.app.state.metrics.retry.labels(reason=exc.category.value).inc()
    request.app.state.metrics.requests.labels(unified.tenant_id, model, "error").inc()
    if last_error is not None:
        raise last_error
    raise GatewayError("no deployment available", ErrorCategory.RETRYABLE_UPSTREAM)


def _pick_attempt(
    request: Request, group: ModelGroupConfig, exclude: set[tuple[str, str]]
) -> tuple[DeploymentConfig, Provider]:
    deployment = pick(group, exclude)
    provider = request.app.state.provider_manager.providers[deployment.provider]
    return deployment, provider


class _QuotaSettlement:
    def __init__(
        self,
        request: Request,
        reservation: QuotaReservation,
        quota: QuotaService,
    ) -> None:
        self._request = request
        self._reservation = reservation
        self._quota = quota
        self._settled = False
        self._usage: Usage | None = None

    def capture_usage(self, usage: Usage | None) -> None:
        self._usage = usage

    async def settle_once(self, usage: Usage | None, forwarded_tokens: int) -> None:
        if self._settled:
            return
        self._settled = True
        final_usage = usage or self._usage
        await self._quota.settle(self._reservation, final_usage, forwarded_tokens)
        _record_quota_metrics(self._request, self._reservation, final_usage)


async def _reserve_quota(
    *,
    request: Request,
    deployment: DeploymentConfig,
    unified: UnifiedRequest,
    snapshot: GatewayConfig,
    count_request_rejection: bool = True,
) -> _QuotaSettlement:
    tenant: TenantConfig = request.state.tenant
    quota: QuotaService = request.app.state.quota
    try:
        reservation = await quota.reserve(
            tenant=tenant,
            req=unified,
            deployment=deployment,
            snapshot=snapshot,
        )
    except GatewayError as exc:
        if exc.category == ErrorCategory.QUOTA_EXCEEDED:
            dim = (exc.code or "quota_exceeded").removesuffix("_exceeded")
            request.app.state.metrics.quota_rejections.labels(tenant.id, dim).inc()
            if count_request_rejection:
                request.app.state.metrics.requests.labels(
                    tenant.id,
                    unified.model,
                    "rejected",
                ).inc()
        raise
    return _QuotaSettlement(request, reservation, quota)


def _record_quota_metrics(
    request: Request,
    reservation: QuotaReservation,
    usage: Usage | None,
) -> None:
    if usage is None:
        return
    metrics = request.app.state.metrics
    metrics.tokens.labels(reservation.tenant_id, reservation.model, "in").inc(usage.prompt_tokens)
    metrics.tokens.labels(reservation.tenant_id, reservation.model, "out").inc(
        usage.completion_tokens
    )
    cost_micro = int(
        usage.prompt_tokens / 1000 * reservation.deployment.price_per_1k_input_usd * 1_000_000
        + usage.completion_tokens
        / 1000
        * reservation.deployment.price_per_1k_output_usd
        * 1_000_000
    )
    metrics.cost.labels(reservation.tenant_id, reservation.model).inc(cost_micro)


def _deadline(settings: GatewayConfig) -> Deadline:
    loop = asyncio.get_running_loop()
    return Deadline(
        connect_s=settings.timeouts.connect_s,
        ttft_s=settings.timeouts.ttft_s,
        inter_chunk_s=settings.timeouts.inter_chunk_s,
        total_deadline=loop.time() + settings.timeouts.total_s,
    )


def _unified_request(
    request: Request,
    incoming: ChatCompletionIn,
    raw_body: dict[str, Any],
    tenant: TenantConfig,
    request_id: str,
) -> UnifiedRequest:
    body = dict(raw_body)
    mock_directive = request.headers.get("x-mock-directive")
    if mock_directive is not None:
        # DECISION: M0 keeps deterministic mock tests header-driven without public API.
        body["__tidegate_mock_directive"] = mock_directive
    prompt_version = request.headers.get("X-Prompt-Version", "default")
    return UnifiedRequest(
        request_id=request_id,
        tenant_id=tenant.id,
        model=incoming.model,
        messages=incoming.messages,
        stream=incoming.stream,
        temperature=incoming.temperature,
        top_p=incoming.top_p,
        max_tokens=incoming.max_tokens,
        stop=incoming.normalized_stop(),
        logprobs=incoming.logprobs,
        prompt_version=prompt_version,
        has_tools=incoming.tools is not None,
        raw_body=body,
    )


def _accumulate(accounting: StreamAccounting, delta: UnifiedDelta) -> None:
    if delta.content:
        accounting.content_chars += len(delta.content)
        accounting.delta_count += 1
    if delta.usage is not None:
        accounting.usage = delta.usage


def _non_stream_payload(
    request_id: str,
    model: str,
    response: UnifiedResponse,
) -> dict[str, object]:
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response.content},
                "finish_reason": response.finish_reason,
            },
        ],
        "usage": response.usage.model_dump(),
    }
