from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, cast

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from opentelemetry.util.types import Attributes
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
from tidegate.cache.gates import read_decision
from tidegate.cache.keys import exact_key
from tidegate.cache.normalize import l1_digest
from tidegate.cache.replay import replay_as_stream
from tidegate.cache.service import CacheHit, CacheService
from tidegate.cache.singleflight import Flight
from tidegate.config.models import DeploymentConfig, GatewayConfig, TenantConfig
from tidegate.core.deadline import Deadline
from tidegate.core.errors import ErrorCategory, GatewayError
from tidegate.core.models import (
    ChatCompletionIn,
    UnifiedDelta,
    UnifiedRequest,
    UnifiedResponse,
    Usage,
)
from tidegate.obs.otel import current_trace_id, start_span
from tidegate.providers.base import Provider
from tidegate.quota.service import QuotaReservation, QuotaService
from tidegate.routing.cascade import cascade_decision, draft_accepted, force_logprobs
from tidegate.routing.hedge import HedgeBudget, trigger_delay_s
from tidegate.routing.ladder import RouteLevel, RoutingLadder
from tidegate.routing.selector import NoAvailableDeployment, P2CSelector
from tidegate.routing.stats import RoutingState
from tidegate.settlement import LedgerBatcher, LedgerRecord

router = APIRouter()


class _ClientDisconnectedError(Exception):
    pass


@dataclass(frozen=True)
class _PickedAttempt:
    deployment: DeploymentConfig
    provider: Provider
    model_group_name: str
    degraded: str | None


@dataclass(frozen=True)
class _PickResult:
    attempt: _PickedAttempt
    level_index: int


@dataclass(frozen=True)
class _ChatResult:
    response: UnifiedResponse
    route_header: str
    cache_header: str
    degraded: str | None = None


@dataclass
class _StreamAttemptState:
    cacheable_content: list[str]
    finish_reason: str | None = None


@dataclass
class _FirstDelta:
    delta: UnifiedDelta | None
    upstream: AsyncIterator[UnifiedDelta]
    provider_name: str
    deployment: DeploymentConfig
    is_hedge: bool


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


@router.post("/v1/cache/feedback")
async def cache_feedback(request: Request) -> JSONResponse:
    payload = await request.json()
    request_id = payload.get("request_id")
    verdict = payload.get("verdict")
    if not isinstance(request_id, str) or verdict != "wrong_answer":
        raise GatewayError("request validation failed", ErrorCategory.CLIENT_ERROR, http_status=422)
    evicted = await _cache_service(request).evict_feedback(request_id)
    return JSONResponse({"ok": True, "evicted": evicted})


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

    if incoming.model not in settings.model_groups:
        raise GatewayError("unknown model", ErrorCategory.CLIENT_ERROR, http_status=404)
    tenant: TenantConfig = request.state.tenant
    request_id: str = request.state.request_id
    deadline = _deadline(settings)
    unified = _unified_request(request, incoming, raw_body, tenant, request_id)
    # REWORK-M0-1: gateway overhead is request receipt through upstream dispatch.
    request.app.state.metrics.overhead.observe(time.monotonic() - request_started)
    levels = RoutingLadder(settings).levels(incoming.model, tenant)
    if incoming.stream:
        policy = settings.policies[tenant.policy]
        if policy.cascade.enabled:
            request.app.state.metrics.cascade.labels("bypassed").inc()
        stream_exclude: set[tuple[str, str]] = set()
        picked = _pick_next_attempt(request, levels, stream_exclude, 0)
        deployment = picked.attempt.deployment
        stream_exclude.add((deployment.provider, deployment.upstream_model))
        quota_settle = await _reserve_quota(
            request=request,
            deployment=deployment,
            unified=unified,
            snapshot=settings,
        )
        cache_header = "bypass" if _cache_bypassed(unified, tenant, settings) else "miss"
        cache_hit = None
        flight: Flight[UnifiedResponse] | None = None
        if cache_header != "bypass":
            cache_hit = await _lookup_cache_with_spans(request, unified, tenant, settings)
        if cache_hit is not None:
            await quota_settle.refund_full()
            _cache_service(request).remember_request_entry(
                unified.request_id,
                cache_hit.semcache_entry_id,
            )
            return DisconnectAwareStreamingResponse(
                _stream_cached_response(request, incoming, cache_hit.response),
                media_type="text/event-stream",
                headers={
                    "X-TideGate-Cache": cache_hit.cache_header,
                    "X-TideGate-Route": "cache",
                },
            )
        if cache_header != "bypass":
            flight_key = exact_key(tenant.id, l1_digest(unified))
            flight = _cache_service(request).acquire(flight_key)
            if not flight.leader:
                try:
                    response = await asyncio.wait_for(
                        asyncio.shield(flight.future),
                        timeout=deadline.remaining(),
                    )
                    await quota_settle.refund_full()
                    return DisconnectAwareStreamingResponse(
                        _stream_cached_response(request, incoming, response),
                        media_type="text/event-stream",
                        headers={
                            "X-TideGate-Cache": "hit-exact",
                            "X-TideGate-Route": "cache",
                        },
                    )
                except Exception:
                    # SPEC-M4-3: stream followers also fall back to an independent upstream
                    # stream when the leader fails instead of inheriting the leader error.
                    pass
            else:
                cache_hit = await _lookup_cache_with_spans(request, unified, tenant, settings)
                if cache_hit is not None:
                    await quota_settle.refund_full()
                    _cache_service(request).remember_request_entry(
                        unified.request_id,
                        cache_hit.semcache_entry_id,
                    )
                    _cache_service(request).resolve(flight, cache_hit.response)
                    _cache_service(request).release(flight)
                    return DisconnectAwareStreamingResponse(
                        _stream_cached_response(request, incoming, cache_hit.response),
                        media_type="text/event-stream",
                        headers={
                            "X-TideGate-Cache": cache_hit.cache_header,
                            "X-TideGate-Route": "cache",
                        },
                    )
        return DisconnectAwareStreamingResponse(
            _stream_with_retries(
                request=request,
                unified=unified,
                deadline=deadline,
                incoming=incoming,
                levels=levels,
                first_attempt=(picked.attempt, quota_settle),
                exclude=stream_exclude,
                level_index=picked.level_index,
                flight=flight,
            ),
            media_type="text/event-stream",
            headers={
                "X-TideGate-Cache": cache_header,
                "X-TideGate-Route": _route_header(picked.attempt.deployment),
                **_degraded_header(picked.attempt.degraded),
            },
        )

    started = time.monotonic()
    result = await _call_non_stream_with_retries(
        request=request,
        levels=levels,
        unified=unified,
        deadline=deadline,
        model=incoming.model,
    )
    duration = time.monotonic() - started
    response = result.response
    outcome = _outcome_for(result)
    request.app.state.metrics.requests.labels(tenant.id, incoming.model, outcome).inc()
    structlog.get_logger().info(
        "access",
        tenant=tenant.id,
        model=incoming.model,
        outcome=outcome,
        route=result.route_header,
        ttft_ms=None,
        duration_ms=duration * 1000,
        forwarded_chars=len(response.content),
        forwarded_deltas=1,
        usage=response.usage.model_dump(),
        trace_id=current_trace_id(),
    )
    return JSONResponse(
        _non_stream_payload(request_id, incoming.model, response),
        headers={
            "X-TideGate-Cache": result.cache_header,
            "X-TideGate-Route": result.route_header,
            **_degraded_header(result.degraded),
        },
    )


async def _render_stream(
    *,
    request: Request,
    upstream: AsyncIterator[UnifiedDelta],
    incoming: ChatCompletionIn,
    provider_name: str,
    upstream_model: str,
    is_hedge: bool,
    accounting: StreamAccounting,
    quota_settle: _QuotaSettlement,
    attempt_started_at: float,
    stream_state: _StreamAttemptState,
) -> AsyncIterator[bytes]:
    settings: GatewayConfig = request.app.state.config_holder.current
    request_id: str = request.state.request_id
    try:
        span_attrs: Attributes = {
            "provider": provider_name,
            "upstream_model": upstream_model,
            "hedge": is_hedge,
        }
        with start_span("upstream.attempt#stream", span_attrs):
            async for delta in with_heartbeats(upstream, settings.server.sse_heartbeat_interval_s):
                async for chunk in _render_stream_delta(
                    request=request,
                    incoming=incoming,
                    delta=delta,
                    provider_name=provider_name,
                    accounting=accounting,
                    attempt_started_at=attempt_started_at,
                    stream_state=stream_state,
                    request_id=request_id,
                ):
                    yield chunk
        quota_settle.capture_usage(accounting.usage)
    except asyncio.CancelledError:
        request.app.state.metrics.upstream_aborted.labels(
            provider=provider_name, reason="client_disconnect"
        ).inc()
        raise


async def _render_stream_delta(
    *,
    request: Request,
    incoming: ChatCompletionIn,
    delta: UnifiedDelta | None,
    provider_name: str,
    accounting: StreamAccounting,
    attempt_started_at: float,
    stream_state: _StreamAttemptState,
    request_id: str,
) -> AsyncIterator[bytes]:
    if await request.is_disconnected():
        # SPEC-M0-5: client disconnects close upstream streaming immediately.
        request.app.state.metrics.upstream_aborted.labels(
            provider=provider_name, reason="client_disconnect"
        ).inc()
        raise _ClientDisconnectedError
    if delta is None:
        yield heartbeat_event()
        return
    if delta.content:
        stream_state.cacheable_content.append(delta.content)
    if delta.finish_reason is not None:
        stream_state.finish_reason = delta.finish_reason
    now = time.monotonic()
    if accounting.ttft_ms is None and delta.content:
        accounting.ttft_ms = (now - attempt_started_at) * 1000
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


async def _render_stream_with_hedge(
    *,
    request: Request,
    unified: UnifiedRequest,
    primary: _PickedAttempt,
    levels: list[RouteLevel],
    attempted: set[tuple[str, str]],
    incoming: ChatCompletionIn,
    deadline: Deadline,
    accounting: StreamAccounting,
    quota_settle: _QuotaSettlement,
    attempt_started_at: float,
    stream_state: _StreamAttemptState,
) -> tuple[AsyncIterator[bytes], DeploymentConfig, bool]:
    settings: GatewayConfig = request.app.state.config_holder.current
    tenant: TenantConfig = request.state.tenant
    policy = settings.policies[tenant.policy]
    hedge_budget = cast(HedgeBudget, getattr(request.app.state, "hedge_budget", HedgeBudget()))
    hedge_budget.record_request()
    primary_upstream = primary.provider.stream_chat(
        unified,
        primary.deployment.upstream_model,
        deadline,
    )
    if not policy.hedging.enabled:
        return (
            _render_stream(
                request=request,
                upstream=primary_upstream,
                incoming=incoming,
                provider_name=primary.deployment.provider,
                upstream_model=primary.deployment.upstream_model,
                is_hedge=False,
                accounting=accounting,
                quota_settle=quota_settle,
                attempt_started_at=attempt_started_at,
                stream_state=stream_state,
            ),
            primary.deployment,
            False,
        )
    if not hedge_budget.allow(policy.hedging):
        request.app.state.metrics.hedge.labels("skipped_budget").inc()
        return (
            _render_stream(
                request=request,
                upstream=primary_upstream,
                incoming=incoming,
                provider_name=primary.deployment.provider,
                upstream_model=primary.deployment.upstream_model,
                is_hedge=False,
                accounting=accounting,
                quota_settle=quota_settle,
                attempt_started_at=attempt_started_at,
                stream_state=stream_state,
            ),
            primary.deployment,
            False,
        )
    delay_s = trigger_delay_s(primary.deployment, _routing_state(request), policy.hedging)
    picked_hedge: _PickedAttempt | None = None
    hedge_exclude = set(attempted)
    hedge_exclude.add((primary.deployment.provider, primary.deployment.upstream_model))
    try:
        picked_hedge = _pick_next_attempt(request, levels, hedge_exclude, 0).attempt
    except GatewayError:
        picked_hedge = None
    if picked_hedge is None:
        return (
            _render_stream(
                request=request,
                upstream=primary_upstream,
                incoming=incoming,
                provider_name=primary.deployment.provider,
                upstream_model=primary.deployment.upstream_model,
                is_hedge=False,
                accounting=accounting,
                quota_settle=quota_settle,
                attempt_started_at=attempt_started_at,
                stream_state=stream_state,
            ),
            primary.deployment,
            False,
        )
    first = await _first_delta_with_optional_hedge(
        primary=_FirstDelta(
            delta=None,
            upstream=primary_upstream,
            provider_name=primary.deployment.provider,
            deployment=primary.deployment,
            is_hedge=False,
        ),
        hedge=picked_hedge,
        delay_s=delay_s,
        unified=unified,
        deadline=deadline,
    )
    if first.is_hedge:
        request.app.state.metrics.hedge.labels("won").inc()
        request.app.state.metrics.upstream_aborted.labels(
            provider=primary.deployment.provider,
            reason="hedge_loser",
        ).inc()
    else:
        request.app.state.metrics.hedge.labels("lost").inc()
        request.app.state.metrics.upstream_aborted.labels(
            provider=picked_hedge.deployment.provider,
            reason="hedge_loser",
        ).inc()
    return (
        _render_stream_from_first(
            request=request,
            first=first,
            incoming=incoming,
            accounting=accounting,
            quota_settle=quota_settle,
            attempt_started_at=attempt_started_at,
            stream_state=stream_state,
        ),
        first.deployment,
        first.is_hedge,
    )


async def _first_delta_with_optional_hedge(
    *,
    primary: _FirstDelta,
    hedge: _PickedAttempt,
    delay_s: float,
    unified: UnifiedRequest,
    deadline: Deadline,
) -> _FirstDelta:
    primary_task = asyncio.create_task(_next_upstream_delta(primary.upstream))
    done, _ = await asyncio.wait({primary_task}, timeout=delay_s)
    if primary_task in done:
        primary.delta = primary_task.result()
        return primary
    hedge_upstream = hedge.provider.stream_chat(unified, hedge.deployment.upstream_model, deadline)
    hedge_task = asyncio.create_task(_next_upstream_delta(hedge_upstream))
    done, pending = await asyncio.wait(
        {primary_task, hedge_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    winner = next(iter(done))
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    if winner is primary_task:
        await _close_iterator(hedge_upstream)
        primary.delta = winner.result()
        return primary
    await _close_iterator(primary.upstream)
    return _FirstDelta(
        delta=winner.result(),
        upstream=hedge_upstream,
        provider_name=hedge.deployment.provider,
        deployment=hedge.deployment,
        is_hedge=True,
    )


async def _render_stream_from_first(
    *,
    request: Request,
    first: _FirstDelta,
    incoming: ChatCompletionIn,
    accounting: StreamAccounting,
    quota_settle: _QuotaSettlement,
    attempt_started_at: float,
    stream_state: _StreamAttemptState,
) -> AsyncIterator[bytes]:
    async def chain() -> AsyncIterator[UnifiedDelta]:
        if first.delta is not None:
            yield first.delta
        async for delta in first.upstream:
            yield delta

    async for chunk in _render_stream(
        request=request,
        upstream=chain(),
        incoming=incoming,
        provider_name=first.provider_name,
        upstream_model=first.deployment.upstream_model,
        is_hedge=first.is_hedge,
        accounting=accounting,
        quota_settle=quota_settle,
        attempt_started_at=attempt_started_at,
        stream_state=stream_state,
    ):
        yield chunk


async def _close_iterator(iterator: AsyncIterator[UnifiedDelta]) -> None:
    aclose = getattr(iterator, "aclose", None)
    if callable(aclose):
        await aclose()


async def _next_upstream_delta(upstream: AsyncIterator[UnifiedDelta]) -> UnifiedDelta:
    return await anext(upstream)


async def _stream_cached_response(
    request: Request,
    incoming: ChatCompletionIn,
    response: UnifiedResponse,
) -> AsyncIterator[bytes]:
    settings: GatewayConfig = request.app.state.config_holder.current
    request_id: str = request.state.request_id
    async for delta in replay_as_stream(response, settings):
        payload = stream_chunk_payload(
            request_id,
            incoming.model,
            delta,
            include_usage=incoming.include_stream_usage(),
        )
        if payload is not None:
            yield sse_event(payload)
    yield sse_event("[DONE]")


async def _stream_with_retries(
    *,
    request: Request,
    unified: UnifiedRequest,
    deadline: Deadline,
    incoming: ChatCompletionIn,
    levels: list[RouteLevel],
    first_attempt: tuple[_PickedAttempt, _QuotaSettlement] | None = None,
    exclude: set[tuple[str, str]] | None = None,
    level_index: int = 0,
    flight: Flight[UnifiedResponse] | None = None,
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
    routing_state = _routing_state(request)
    try:
        for attempt_index in range(attempts):
            attempt_count = attempt_index + 1
            if attempt_index == 0 and first_attempt is not None:
                picked, quota_settle = first_attempt
            else:
                try:
                    result = _pick_next_attempt(request, levels, attempted, level_index)
                    picked = result.attempt
                    level_index = result.level_index
                    if quota_settle is None:
                        quota_settle = await _reserve_quota(
                            request=request,
                            deployment=picked.deployment,
                            unified=unified,
                            snapshot=settings,
                        )
                except GatewayError as exc:
                    last_error = exc
                    break
                attempted.add((picked.deployment.provider, picked.deployment.upstream_model))
            deployment = picked.deployment
            if quota_settle is not None:
                quota_settle.use_deployment(deployment)
            else:
                last_error = GatewayError("quota reservation missing", ErrorCategory.INTERNAL)
                break
            route_header = _route_header(deployment)
            attempt_started_at = time.monotonic()
            stream_state = _StreamAttemptState(cacheable_content=[])
            try:
                routing_state.record_start(deployment, now_s=time.monotonic())
            except GatewayError as exc:
                last_error = exc
                continue
            sent_data = False
            try:
                body_iterator, winning_deployment, _ = await _render_stream_with_hedge(
                    request=request,
                    unified=unified,
                    primary=picked,
                    levels=levels,
                    attempted=attempted,
                    incoming=incoming,
                    deadline=deadline,
                    accounting=accounting,
                    quota_settle=quota_settle,
                    attempt_started_at=attempt_started_at,
                    stream_state=stream_state,
                )
                deployment = winning_deployment
                route_header = _route_header(deployment)
                quota_settle.use_deployment(deployment)
                async for chunk in body_iterator:
                    # DECISION: REWORK-M1-1 treats only SSE data events as the idempotency
                    # boundary; heartbeat comments can be followed by a safe TTFT retry.
                    if chunk.startswith(b"data:"):
                        sent_data = True
                    yield chunk
                routing_state.record_finish(
                    deployment,
                    success=True,
                    ttft_s=_ttft_s(accounting),
                    now_s=time.monotonic(),
                )
                outcome = "ok"
                if picked.degraded is not None:
                    outcome = "degraded"
                if accounting.usage is not None:
                    response = UnifiedResponse(
                        content="".join(stream_state.cacheable_content),
                        finish_reason=stream_state.finish_reason or "stop",
                        usage=accounting.usage,
                        model=deployment.upstream_model,
                    )
                    if flight is not None and flight.leader:
                        _cache_service(request).resolve(flight, response)
                    await _cache_service(request).store(
                        unified,
                        tenant,
                        response,
                        settings,
                        degraded=picked.degraded is not None,
                    )
                elif flight is not None and flight.leader:
                    _cache_service(request).reject(
                        flight,
                        GatewayError(
                            "stream response missing usage",
                            ErrorCategory.RETRYABLE_UPSTREAM,
                        ),
                    )
                yield sse_event("[DONE]")
                return
            except _ClientDisconnectedError:
                outcome = "client_disconnect"
                routing_state.record_abort(deployment)
                if flight is not None and flight.leader:
                    _cache_service(request).reject(flight, _ClientDisconnectedError())
                return
            except GatewayError as exc:
                last_error = exc
                _record_route_error(
                    request,
                    deployment,
                    exc,
                    ttft_s=_ttft_s(accounting),
                    now_s=time.monotonic(),
                )
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
                continue
        if last_error is None:
            last_error = GatewayError("no deployment available", ErrorCategory.RETRYABLE_UPSTREAM)
        stale_hit = await RoutingLadder(settings).stale_cache(
            _cache_service(request),
            unified,
            tenant,
        )
        if stale_hit is not None:
            if quota_settle is not None:
                await quota_settle.refund_full()
            _cache_service(request).remember_request_entry(
                unified.request_id,
                stale_hit.semcache_entry_id,
            )
            # SPEC-M4-7: streaming headers were already sent at admission time, so
            # stale-cache degradation is only visible through the replayed SSE body.
            outcome = "degraded"
            route_header = "cache"
            if flight is not None and flight.leader:
                _cache_service(request).reject(
                    flight,
                    GatewayError("leader served stale cache", ErrorCategory.RETRYABLE_UPSTREAM),
                )
            async for chunk in _stream_cached_response(request, incoming, stale_hit.response):
                yield chunk
            return
        # REWORK-M1-2: once StreamingResponse has been selected, every failure is in-band.
        structlog.get_logger().error(
            "stream_request_failed",
            category=last_error.category.value,
            attempts=attempt_count,
            route=route_header,
        )
        if flight is not None and flight.leader:
            _cache_service(request).reject(flight, last_error)
        async for chunk in error_chunk(unified.request_id, incoming.model):
            yield chunk
    except asyncio.CancelledError:
        outcome = "client_disconnect"
        if flight is not None and flight.leader:
            _cache_service(request).reject(flight, _ClientDisconnectedError())
        raise
    except BaseException as exc:
        if flight is not None and flight.leader:
            _cache_service(request).reject(flight, exc)
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
            trace_id=current_trace_id(),
        )
        if quota_settle is not None:
            await quota_settle.settle_once(accounting.usage, accounting.delta_count)
        if accounting.usage is not None:
            _enqueue_ledger(
                request,
                unified=unified,
                usage=accounting.usage,
                route_header=route_header,
                cache_header="miss",
                degraded=None if outcome != "degraded" else "stream-degraded",
                outcome=outcome,
                settings=settings,
            )
        if flight is not None and flight.leader:
            _cache_service(request).release(flight)


async def _call_non_stream_with_retries(
    *,
    request: Request,
    levels: list[RouteLevel],
    unified: UnifiedRequest,
    deadline: Deadline,
    model: str,
) -> _ChatResult:
    settings: GatewayConfig = request.app.state.config_holder.current
    tenant: TenantConfig = request.state.tenant
    cache = _cache_service(request)
    initial_deployment = levels[0].group.deployments[0]
    bypass_cache = _cache_bypassed(unified, tenant, settings)
    cache_quota = await _reserve_quota(
        request=request,
        deployment=initial_deployment,
        unified=unified,
        snapshot=settings,
    )
    cache_hit = (
        None if bypass_cache else await _lookup_cache_with_spans(request, unified, tenant, settings)
    )
    if cache_hit is not None:
        await cache_quota.refund_full()
        cache.remember_request_entry(unified.request_id, cache_hit.semcache_entry_id)
        _enqueue_ledger(
            request,
            unified=unified,
            usage=cache_hit.response.usage,
            route_header="cache",
            cache_header=cache_hit.cache_header,
            degraded=None,
            outcome="cached",
            settings=settings,
        )
        return _ChatResult(cache_hit.response, "cache", cache_hit.cache_header)
    if bypass_cache:
        return await _call_non_stream_with_cascade(
            request=request,
            levels=levels,
            unified=unified,
            deadline=deadline,
            model=model,
            cache=cache,
            cache_quota=cache_quota,
            flight=None,
            cache_header="bypass",
        )

    flight_key = exact_key(tenant.id, l1_digest(unified))
    flight = cache.acquire(flight_key)
    if not flight.leader:
        try:
            response = await asyncio.wait_for(
                asyncio.shield(flight.future),
                timeout=deadline.remaining(),
            )
            await cache_quota.refund_full()
            _enqueue_ledger(
                request,
                unified=unified,
                usage=response.usage,
                route_header="cache",
                cache_header="hit-exact",
                degraded=None,
                outcome="cached",
                settings=settings,
            )
            return _ChatResult(response, "cache", "hit-exact")
        except Exception:
            # SPEC-M4-3: follower fallback is an independent upstream attempt when leader fails.
            pass
    else:
        # SPEC-M4-3: a request can miss L1 before the previous leader stores, then acquire
        # a fresh leader slot after that leader releases. Re-check before calling upstream
        # so the same concurrent miss still collapses to one provider request.
        cache_hit = await _lookup_cache_with_spans(request, unified, tenant, settings)
        if cache_hit is not None:
            await cache_quota.refund_full()
            cache.remember_request_entry(unified.request_id, cache_hit.semcache_entry_id)
            cache.resolve(flight, cache_hit.response)
            cache.release(flight)
            _enqueue_ledger(
                request,
                unified=unified,
                usage=cache_hit.response.usage,
                route_header="cache",
                cache_header=cache_hit.cache_header,
                degraded=None,
                outcome="cached",
                settings=settings,
            )
            return _ChatResult(cache_hit.response, "cache", cache_hit.cache_header)

    return await _call_non_stream_with_cascade(
        request=request,
        levels=levels,
        unified=unified,
        deadline=deadline,
        model=model,
        cache=cache,
        cache_quota=cache_quota,
        flight=flight,
        cache_header="miss",
    )


async def _call_non_stream_with_cascade(
    *,
    request: Request,
    levels: list[RouteLevel],
    unified: UnifiedRequest,
    deadline: Deadline,
    model: str,
    cache: CacheService,
    cache_quota: _QuotaSettlement,
    flight: Flight[UnifiedResponse] | None,
    cache_header: str,
) -> _ChatResult:
    settings: GatewayConfig = request.app.state.config_holder.current
    tenant: TenantConfig = request.state.tenant
    policy = settings.policies[tenant.policy]
    decision = cascade_decision(unified, policy, settings)
    if not decision.enabled:
        if decision.reason == "stream":
            request.app.state.metrics.cascade.labels("bypassed").inc()
        return await _call_non_stream_upstream(
            request=request,
            levels=levels,
            unified=unified,
            deadline=deadline,
            model=model,
            cache=cache,
            cache_quota=cache_quota,
            flight=flight,
            cache_header=cache_header,
        )
    draft_levels = RoutingLadder(settings).levels(decision.draft_group or model, tenant)
    draft = await _call_non_stream_upstream(
        request=request,
        levels=draft_levels,
        unified=force_logprobs(unified),
        deadline=deadline,
        model=model,
        cache=cache,
        cache_quota=cache_quota,
        flight=None,
        cache_header=cache_header,
        store_cache=False,
    )
    if draft_accepted(draft.response, policy.cascade):
        request.app.state.metrics.cascade.labels("draft_accepted").inc()
        if flight is not None and flight.leader:
            cache.reject(
                flight,
                GatewayError("cascade draft accepted", ErrorCategory.RETRYABLE_UPSTREAM),
            )
            cache.release(flight)
        return draft
    request.app.state.metrics.cascade.labels("escalated").inc()
    return await _call_non_stream_upstream(
        request=request,
        levels=levels,
        unified=unified,
        deadline=deadline,
        model=model,
        cache=cache,
        cache_quota=None,
        flight=flight,
        cache_header=cache_header,
    )


async def _call_non_stream_upstream(
    *,
    request: Request,
    levels: list[RouteLevel],
    unified: UnifiedRequest,
    deadline: Deadline,
    model: str,
    cache: CacheService,
    cache_quota: _QuotaSettlement | None,
    flight: Flight[UnifiedResponse] | None,
    cache_header: str,
    store_cache: bool = True,
) -> _ChatResult:
    settings: GatewayConfig = request.app.state.config_holder.current
    tenant: TenantConfig = request.state.tenant
    result: _ChatResult | None = None
    quota_settle: _QuotaSettlement | None = cache_quota
    exclude: set[tuple[str, str]] = set()
    last_error: GatewayError | None = None
    level_index = 0
    routing_state = _routing_state(request)
    try:
        for _ in range(settings.routing.max_attempts_before_first_byte):
            try:
                pick_result = _pick_next_attempt(request, levels, exclude, level_index)
            except GatewayError as exc:
                last_error = exc
                break
            picked = pick_result.attempt
            level_index = pick_result.level_index
            deployment = picked.deployment
            provider = picked.provider
            exclude.add((deployment.provider, deployment.upstream_model))
            route_header = _route_header(deployment)
            if quota_settle is None:
                quota_settle = await _reserve_quota(
                    request=request,
                    deployment=deployment,
                    unified=unified,
                    snapshot=settings,
                )
            quota_settle.use_deployment(deployment)
            attempt_started_at = time.monotonic()
            try:
                routing_state.record_start(deployment, now_s=time.monotonic())
            except GatewayError as exc:
                last_error = exc
                continue
            try:
                span_attrs: Attributes = {
                    "provider": deployment.provider,
                    "upstream_model": deployment.upstream_model,
                    "hedge": False,
                }
                with start_span("upstream.attempt#non_stream", span_attrs):
                    response = await provider.chat(unified, deployment.upstream_model, deadline)
                await quota_settle.settle_once(response.usage, response.usage.completion_tokens)
                routing_state.record_finish(
                    deployment,
                    success=True,
                    ttft_s=time.monotonic() - attempt_started_at,
                    now_s=time.monotonic(),
                )
                result = _ChatResult(response, route_header, cache_header, picked.degraded)
                _enqueue_ledger(
                    request,
                    unified=unified,
                    usage=response.usage,
                    route_header=route_header,
                    cache_header=cache_header,
                    degraded=picked.degraded,
                    outcome="degraded" if picked.degraded is not None else "ok",
                    settings=settings,
                )
                if store_cache:
                    await cache.store(
                        unified,
                        tenant,
                        response,
                        settings,
                        degraded=picked.degraded is not None,
                    )
                if flight is not None and flight.leader:
                    cache.resolve(flight, response)
                quota_settle = None
                return result
            except GatewayError as exc:
                if quota_settle is None:
                    raise GatewayError("quota settlement missing", ErrorCategory.INTERNAL) from exc
                await quota_settle.settle_once(None, 0)
                quota_settle = None
                last_error = exc
                _record_route_error(
                    request,
                    deployment,
                    exc,
                    ttft_s=time.monotonic() - attempt_started_at,
                    now_s=time.monotonic(),
                )
                if exc.category not in {
                    ErrorCategory.RETRYABLE_UPSTREAM,
                    ErrorCategory.RATE_LIMITED_UPSTREAM,
                    ErrorCategory.TIMEOUT_TTFT,
                }:
                    request.app.state.metrics.requests.labels(
                        unified.tenant_id, model, "error"
                    ).inc()
                    raise
                request.app.state.metrics.retry.labels(reason=exc.category.value).inc()
        stale_hit = await RoutingLadder(settings).stale_cache(cache, unified, tenant)
        if stale_hit is not None:
            if quota_settle is not None:
                await quota_settle.refund_full()
                quota_settle = None
            cache.remember_request_entry(unified.request_id, stale_hit.semcache_entry_id)
            if flight is not None and flight.leader:
                cache.reject(
                    flight,
                    GatewayError("leader served stale cache", ErrorCategory.RETRYABLE_UPSTREAM),
                )
            _enqueue_ledger(
                request,
                unified=unified,
                usage=stale_hit.response.usage,
                route_header="cache",
                cache_header=stale_hit.cache_header,
                degraded="stale-cache",
                outcome="degraded",
                settings=settings,
            )
            return _ChatResult(stale_hit.response, "cache", stale_hit.cache_header, "stale-cache")
        if quota_settle is not None:
            await quota_settle.settle_once(None, 0)
            quota_settle = None
        request.app.state.metrics.requests.labels(unified.tenant_id, model, "error").inc()
        if last_error is not None:
            raise last_error
        raise GatewayError("no deployment available", ErrorCategory.RETRYABLE_UPSTREAM)
    except BaseException as exc:
        if flight is not None and flight.leader and result is None:
            cache.reject(flight, exc)
        raise
    finally:
        if flight is not None and flight.leader:
            cache.release(flight)


def _pick_next_attempt(
    request: Request,
    levels: list[RouteLevel],
    exclude: set[tuple[str, str]],
    level_index: int,
) -> _PickResult:
    last_error: GatewayError | None = None
    for index in range(level_index, len(levels)):
        level = levels[index]
        try:
            attempt = _pick_attempt(request, level, exclude)
        except NoAvailableDeployment as exc:
            last_error = exc
            continue
        return _PickResult(attempt, index)
    if last_error is not None:
        raise last_error
    raise NoAvailableDeployment()


def _pick_attempt(
    request: Request,
    level: RouteLevel,
    exclude: set[tuple[str, str]],
) -> _PickedAttempt:
    selector: P2CSelector | None = getattr(request.app.state, "selector", None)
    if selector is None:
        selector = P2CSelector(
            request.app.state.config_holder.current,
            _routing_state(request),
        )
    with start_span("route.select", {"model_group": level.model_group_name}):
        deployment = selector.pick(
            level.group,
            exclude,
            now_s=asyncio.get_running_loop().time(),
        )
    provider = request.app.state.provider_manager.providers[deployment.provider]
    return _PickedAttempt(deployment, provider, level.model_group_name, level.degraded)


def _record_route_error(
    request: Request,
    deployment: DeploymentConfig,
    exc: GatewayError,
    *,
    ttft_s: float | None,
    now_s: float,
) -> None:
    routing_state = _routing_state(request)
    if exc.category == ErrorCategory.RATE_LIMITED_UPSTREAM:
        settings: GatewayConfig = request.app.state.config_holder.current
        retry_after_s = exc.retry_after_s or settings.routing.breaker.open_cooldown_s
        routing_state.record_rate_limit(deployment, retry_after_s, now_s)
        return
    if exc.category in {
        ErrorCategory.RETRYABLE_UPSTREAM,
        ErrorCategory.TIMEOUT_TTFT,
        ErrorCategory.TIMEOUT_STALL,
        ErrorCategory.TIMEOUT_TOTAL,
    }:
        routing_state.record_finish(deployment, success=False, ttft_s=ttft_s, now_s=now_s)
        return
    routing_state.record_abort(deployment)


def _routing_state(request: Request) -> RoutingState:
    return cast(RoutingState, request.app.state.routing_state)


def _ledger(request: Request) -> LedgerBatcher:
    return cast(LedgerBatcher, request.app.state.ledger)


def _cache_service(request: Request) -> CacheService:
    return cast(CacheService, request.app.state.cache)


async def _lookup_cache_with_spans(
    request: Request,
    unified: UnifiedRequest,
    tenant: TenantConfig,
    settings: GatewayConfig,
    *,
    stale: bool = False,
) -> CacheHit | None:
    return await _cache_service(request).lookup(unified, tenant, settings, stale=stale)


def _outcome_for(result: _ChatResult) -> str:
    if result.degraded is not None:
        return "degraded"
    if result.cache_header in {"hit-exact", "hit-semantic"}:
        return "cached"
    return "ok"


def _cache_bypassed(
    req: UnifiedRequest,
    tenant: TenantConfig,
    settings: GatewayConfig,
) -> bool:
    decision = read_decision(req, tenant, settings)
    return decision.bypass_reason is not None or not decision.l1


def _route_header(deployment: DeploymentConfig) -> str:
    return f"{deployment.provider}/{deployment.upstream_model}"


def _enqueue_ledger(
    request: Request,
    *,
    unified: UnifiedRequest,
    usage: Usage,
    route_header: str,
    cache_header: str,
    degraded: str | None,
    outcome: str,
    settings: GatewayConfig,
) -> None:
    if not hasattr(request.app.state, "ledger"):
        return
    provider, upstream_model = _route_parts(route_header)
    deployment = _deployment_for_route(settings, provider, upstream_model)
    _ledger(request).enqueue(
        LedgerRecord(
            request_id=unified.request_id,
            tenant_id=unified.tenant_id,
            model=unified.model,
            provider=provider,
            upstream_model=upstream_model,
            usage=usage,
            cost_microusd=0 if deployment is None else _usage_cost_microusd(usage, deployment),
            cache_status=cache_header,
            route_path=route_header,
            degraded=degraded,
            outcome=outcome,
        )
    )


def _route_parts(route_header: str) -> tuple[str, str]:
    if "/" not in route_header:
        return route_header, route_header
    provider, upstream_model = route_header.split("/", maxsplit=1)
    return provider, upstream_model


def _deployment_for_route(
    settings: GatewayConfig,
    provider: str,
    upstream_model: str,
) -> DeploymentConfig | None:
    for group in settings.model_groups.values():
        for deployment in group.deployments:
            if deployment.provider == provider and deployment.upstream_model == upstream_model:
                return deployment
    return None


def _usage_cost_microusd(usage: Usage, deployment: DeploymentConfig) -> int:
    cost = (
        usage.prompt_tokens / 1000 * deployment.price_per_1k_input_usd
        + usage.completion_tokens / 1000 * deployment.price_per_1k_output_usd
    )
    return max(0, round(cost * 1_000_000))


def _degraded_header(degraded: str | None) -> dict[str, str]:
    if degraded is None:
        return {}
    return {"X-TideGate-Degraded": degraded}


def _ttft_s(accounting: StreamAccounting) -> float | None:
    if accounting.ttft_ms is None:
        return None
    return accounting.ttft_ms / 1000


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

    def use_deployment(self, deployment: DeploymentConfig) -> None:
        self._reservation = self._reservation.with_deployment(deployment)

    async def settle_once(self, usage: Usage | None, forwarded_tokens: int) -> None:
        if self._settled:
            return
        self._settled = True
        final_usage = usage or self._usage
        with start_span("settle", {"tenant": self._reservation.tenant_id}):
            await self._quota.settle(self._reservation, final_usage, forwarded_tokens)
        _record_quota_metrics(self._request, self._reservation, final_usage)

    async def refund_full(self) -> None:
        if self._settled:
            return
        self._settled = True
        with start_span("settle", {"tenant": self._reservation.tenant_id, "refund": True}):
            await self._quota.refund_full(self._reservation)


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
        with start_span("quota.reserve", {"tenant": tenant.id, "model": unified.model}):
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
