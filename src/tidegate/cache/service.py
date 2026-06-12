from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol

import redis.asyncio as redis

from tidegate.cache.gates import can_store, read_decision
from tidegate.cache.keys import exact_key
from tidegate.cache.l2 import L2Cache, SemanticHit
from tidegate.cache.normalize import l1_digest, semantic_text
from tidegate.cache.singleflight import Flight, SingleFlight
from tidegate.config.models import GatewayConfig, TenantConfig
from tidegate.core.models import UnifiedRequest, UnifiedResponse
from tidegate.obs.metrics import Metrics
from tidegate.obs.otel import start_span


@dataclass(frozen=True)
class CacheHit:
    response: UnifiedResponse
    cache_header: str
    semcache_entry_id: str | None = None


class _L1Store(Protocol):
    async def get(self, key: str) -> UnifiedResponse | None: ...

    async def set(self, key: str, response: UnifiedResponse, settings: GatewayConfig) -> None: ...


class _EmbeddingClient(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def rerank(self, pairs: list[tuple[str, str]]) -> list[float]: ...


class CacheService:
    def __init__(
        self,
        l1: _L1Store,
        l2: L2Cache,
        embedding: _EmbeddingClient | None,
        metrics: Metrics,
    ) -> None:
        self._l1 = l1
        self._l2 = l2
        self._embedding = embedding
        self._metrics = metrics
        self._singleflight: SingleFlight[UnifiedResponse] = SingleFlight()
        self._request_entries: OrderedDict[str, str] = OrderedDict()
        self._request_entry_capacity = 10_000

    async def lookup(
        self,
        req: UnifiedRequest,
        tenant: TenantConfig,
        settings: GatewayConfig,
        *,
        stale: bool = False,
    ) -> CacheHit | None:
        decision = read_decision(req, tenant, settings)
        if not decision.l1:
            self._metrics.cache_events.labels("l1", "skip").inc()
            return None
        key = exact_key(tenant.id, l1_digest(req))
        try:
            with start_span("cache.l1", {"stale": stale}):
                response = await self._l1.get(key)
        except redis.RedisError:
            self._metrics.cache_events.labels("l1", "skip").inc()
            return None
        if response is not None:
            self._metrics.cache_events.labels("l1", "hit").inc()
            return CacheHit(response, "hit-exact")
        self._metrics.cache_events.labels("l1", "miss").inc()
        if not decision.l2 or self._embedding is None:
            if decision.l2:
                self._metrics.cache_events.labels("l2", "skip").inc()
            return None
        with start_span("cache.l2", {"stale": stale}):
            reranked = await self._lookup_l2(req, tenant, settings, stale=stale)
        if reranked is None:
            self._metrics.cache_events.labels("l2", "miss").inc()
            return None
        hit, response = reranked
        try:
            await self._l2.mark_hit(hit.entry_id)
        except redis.RedisError:
            self._metrics.cache_events.labels("l2", "skip").inc()
            return None
        self._metrics.cache_events.labels("l2", "hit").inc()
        return CacheHit(response, "hit-semantic", hit.entry_id)

    async def _candidate_response(self, hit: SemanticHit, *, stale: bool) -> UnifiedResponse | None:
        try:
            with start_span("cache.l1", {"stale": stale, "semantic_hit": True}):
                response = await self._l1.get(hit.l1_key)
        except redis.RedisError:
            self._metrics.cache_events.labels("l1", "skip").inc()
            return None
        if response is not None:
            return response
        try:
            await self._l2.delete(hit.entry_id)
        except redis.RedisError:
            self._metrics.cache_events.labels("l2", "skip").inc()
        return None

    async def _responses_for_candidates(
        self,
        candidates: list[SemanticHit],
        *,
        stale: bool,
    ) -> list[tuple[SemanticHit, UnifiedResponse]]:
        usable: list[tuple[SemanticHit, UnifiedResponse]] = []
        for candidate in candidates:
            response = await self._candidate_response(candidate, stale=stale)
            if response is not None:
                usable.append((candidate, response))
        return usable

    def acquire(self, key: str) -> Flight[UnifiedResponse]:
        return self._singleflight.acquire(key)

    def resolve(self, flight: Flight[UnifiedResponse], response: UnifiedResponse) -> None:
        self._singleflight.resolve(flight, response)

    def reject(self, flight: Flight[UnifiedResponse], exc: BaseException) -> None:
        self._singleflight.reject(flight, exc)

    def release(self, flight: Flight[UnifiedResponse]) -> None:
        self._singleflight.release(flight)

    async def store(
        self,
        req: UnifiedRequest,
        tenant: TenantConfig,
        response: UnifiedResponse,
        settings: GatewayConfig,
        *,
        degraded: bool,
    ) -> None:
        store_l1, store_l2 = can_store(req, tenant, response, settings, degraded=degraded)
        if not store_l1:
            self._metrics.cache_events.labels("l1", "skip").inc()
            return
        key = exact_key(tenant.id, l1_digest(req))
        try:
            await self._l1.set(key, response, settings)
        except redis.RedisError:
            self._metrics.cache_events.labels("l1", "skip").inc()
            return
        self._metrics.cache_events.labels("l1", "store").inc()
        if store_l2 and self._embedding is not None:
            try:
                async with asyncio.timeout(settings.cache.l2.store_timeout_ms / 1000):
                    text = semantic_text(req)
                    vector = (await self._embedding.embed([text]))[0]
                    await self._l2.store(
                        tenant_id=tenant.id,
                        prompt_version=req.prompt_version,
                        vector=vector,
                        l1_key=key,
                        text=text,
                        settings=settings,
                    )
            except (TimeoutError, redis.RedisError):
                self._metrics.cache_events.labels("l2", "skip").inc()
                return
            self._metrics.cache_events.labels("l2", "store").inc()

    async def evict_feedback(self, request_id: str) -> bool:
        entry_id = self._request_entries.pop(request_id, None)
        if entry_id is None:
            return False
        await self._l2.delete(entry_id)
        self._metrics.cache_events.labels("l2", "evict_feedback").inc()
        return True

    def remember_request_entry(self, request_id: str, entry_id: str | None) -> None:
        if entry_id is None:
            return
        self._request_entries[request_id] = entry_id
        self._request_entries.move_to_end(request_id)
        while len(self._request_entries) > self._request_entry_capacity:
            self._request_entries.popitem(last=False)

    async def _lookup_l2(
        self,
        req: UnifiedRequest,
        tenant: TenantConfig,
        settings: GatewayConfig,
        *,
        stale: bool,
    ) -> tuple[SemanticHit, UnifiedResponse] | None:
        if self._embedding is None:
            return None
        rerank_threshold = _tenant_l2_threshold(tenant, settings)
        if stale:
            rerank_threshold -= settings.cache.l2.stale_threshold_delta
        try:
            async with asyncio.timeout(settings.cache.l2.query_timeout_ms / 1000):
                query_text = semantic_text(req)
                vector = (await self._embedding.embed([query_text]))[0]
                candidates = await self._l2.lookup(
                    tenant_id=tenant.id,
                    prompt_version=req.prompt_version,
                    vector=vector,
                    threshold=settings.cache.l2.recall_threshold,
                    top_k=settings.cache.l2.recall_top_k,
                    timeout_ms=settings.cache.l2.query_timeout_ms,
                )
                usable = await self._responses_for_candidates(candidates, stale=stale)
                if not usable:
                    return None
                # DECISION: SPEC-F diagnostics showed positive p50=0.796 and negative
                # p90=0.857 for bge-small-zh-v1.5, so cosine only gates recall; reranker
                # owns false-hit budget even though calibrated recall remains data-bound.
                pairs = [(query_text, hit.text) for hit, _ in usable]
                rerank_scores = await self._embedding.rerank(pairs)
                return _best_reranked_hit(
                    usable,
                    rerank_scores,
                    threshold=rerank_threshold,
                )
        except TimeoutError:
            return None


def _best_reranked_hit(
    candidates: list[tuple[SemanticHit, UnifiedResponse]],
    rerank_scores: list[float],
    *,
    threshold: float,
) -> tuple[SemanticHit, UnifiedResponse] | None:
    if len(candidates) != len(rerank_scores):
        return None
    scored = [
        (score, hit, response)
        for (hit, response), score in zip(candidates, rerank_scores, strict=True)
    ]
    if not scored:
        return None
    score, hit, response = max(scored, key=lambda item: item[0])
    # DECISION: SPEC-F-2 moves tenant operating points from bi-encoder cosine to
    # reranker score; recall_threshold stays wide and only gates candidates into rerank.
    if score < threshold:
        return None
    return SemanticHit(
        entry_id=hit.entry_id, l1_key=hit.l1_key, score=score, text=hit.text
    ), response


def _tenant_l2_threshold(tenant: TenantConfig, settings: GatewayConfig) -> float:
    points = settings.cache.l2.operating_points
    if not points:
        return settings.cache.l2.similarity_threshold
    selected_name = tenant.cache.l2_operating_point
    if selected_name is None:
        # DECISION: REWORK-M4-2 defaults to the most conservative point because small
        # embeddings showed weak recall at 1% FPR; false-hit budget is a tenant business
        # decision, so the gateway exposes calibrated curve points instead of one global tau.
        return max(points, key=lambda point: point.tau).tau
    for point in points:
        if point.name == selected_name:
            return point.tau
    return max(points, key=lambda point: point.tau).tau
