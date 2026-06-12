from __future__ import annotations

import asyncio
from typing import Any

import pytest
import redis

from tidegate.cache.keys import exact_key
from tidegate.cache.l2 import L2Cache, SemanticHit, capacity_sweep_loop
from tidegate.cache.normalize import semantic_text
from tidegate.cache.service import CacheService
from tidegate.config.loader import load_config, load_config_dict
from tidegate.config.models import GatewayConfig
from tidegate.core.models import ChatMessage, UnifiedRequest, UnifiedResponse, Usage
from tidegate.obs.metrics import Metrics


@pytest.mark.asyncio
async def test_l2_lookup_timeout_covers_embedding() -> None:
    """SPEC-M4-5."""
    base = load_config("tests/fixtures/gateway-test.yaml")
    settings = base.model_copy(
        update={
            "cache": base.cache.model_copy(
                update={"l2": base.cache.l2.model_copy(update={"query_timeout_ms": 1})}
            )
        }
    )
    tenant = settings.tenants[0].model_copy(
        update={"cache": settings.tenants[0].cache.model_copy(update={"l2": True})}
    )
    service = CacheService(
        _MissingL1(),
        _MissingL2(),
        _SlowEmbedding(),
        Metrics.create(),
    )

    assert await service.lookup(_req(), tenant, settings) is None


@pytest.mark.asyncio
async def test_l2_store_timeout_skips_semantic_write() -> None:
    """SPEC-M4-5."""
    base = load_config("tests/fixtures/gateway-test.yaml")
    settings = base.model_copy(
        update={
            "cache": base.cache.model_copy(
                update={"l2": base.cache.l2.model_copy(update={"store_timeout_ms": 1})}
            )
        }
    )
    tenant = settings.tenants[0].model_copy(
        update={"cache": settings.tenants[0].cache.model_copy(update={"l2": True})}
    )
    l2 = _MissingL2()
    service = CacheService(
        _MemoryL1(),
        l2,
        _SlowEmbedding(),
        Metrics.create(),
    )

    await service.store(_req(), tenant, _resp(), settings, degraded=False)

    assert l2.stores == 0


@pytest.mark.asyncio
async def test_l2_capacity_sweep_survives_redis_error() -> None:
    """SPEC-M4-5."""
    base = load_config("tests/fixtures/gateway-test.yaml")
    settings = base.model_copy(
        update={
            "cache": base.cache.model_copy(
                update={"l2": base.cache.l2.model_copy(update={"capacity_sweep_interval_s": 0.001})}
            )
        }
    )
    l2 = _FailingCapacityL2()
    task = asyncio.create_task(capacity_sweep_loop(l2, lambda: settings))
    try:
        await asyncio.sleep(0.02)
        assert not task.done()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_l2_rerank_threshold_selects_best_candidate() -> None:
    """SPEC-F-2."""
    base = load_config("tests/fixtures/gateway-test.yaml")
    raw = base.model_dump()
    raw["cache"]["l2"]["recall_threshold"] = 0.70
    raw["cache"]["l2"]["recall_top_k"] = 3
    raw["cache"]["l2"]["operating_points"] = [
        {"name": "balanced", "tau": 0.60, "expected_fpr": 0.01, "expected_recall": 0.90}
    ]
    settings = load_config_dict(raw)
    tenant = settings.tenants[0].model_copy(
        update={
            "cache": settings.tenants[0].cache.model_copy(
                update={"l2": True, "l2_operating_point": "balanced"}
            )
        }
    )
    req = _req()
    semantic_key = exact_key(tenant.id, "semantic-candidate")
    l1 = _MemoryL1({semantic_key: _resp(content="reranked")})
    l2 = _CandidateL2(
        [
            SemanticHit("low", semantic_key, score=0.91, text="无关问题"),
            SemanticHit("high", semantic_key, score=0.80, text=semantic_text(req)),
        ]
    )
    service = CacheService(
        l1,
        l2,
        _RerankEmbedding([0.10, 0.90]),
        Metrics.create(),
    )

    hit = await service.lookup(req, tenant, settings)

    assert hit is not None
    assert hit.response.content == "reranked"
    assert hit.semcache_entry_id == "high"
    assert l2.marked == ["high"]


@pytest.mark.asyncio
async def test_l1_lookup_redis_error_is_cache_miss() -> None:
    """SPEC-M4-3."""
    settings = load_config("tests/fixtures/gateway-test.yaml")
    service = CacheService(
        _FailingL1(),
        _MissingL2(),
        None,
        Metrics.create(),
    )

    assert await service.lookup(_req(), settings.tenants[0], settings) is None


@pytest.mark.asyncio
async def test_l1_store_redis_error_does_not_escape() -> None:
    """SPEC-M4-3."""
    settings = load_config("tests/fixtures/gateway-test.yaml")
    service = CacheService(
        _FailingL1(),
        _MissingL2(),
        None,
        Metrics.create(),
    )

    await service.store(_req(), settings.tenants[0], _resp(), settings, degraded=False)


class _MissingL1:
    async def get(self, key: str) -> UnifiedResponse | None:
        del key
        return None

    async def set(self, key: str, response: UnifiedResponse, settings: GatewayConfig) -> None:
        del key, response, settings


class _MemoryL1(_MissingL1):
    def __init__(self, values: dict[str, UnifiedResponse] | None = None) -> None:
        self._values = values or {}

    async def get(self, key: str) -> UnifiedResponse | None:
        return self._values.get(key)

    async def set(self, key: str, response: UnifiedResponse, settings: GatewayConfig) -> None:
        del key, response, settings


class _FailingL1(_MissingL1):
    async def get(self, key: str) -> None:
        del key
        raise redis.RedisError("l1 unavailable")

    async def set(self, key: str, response: UnifiedResponse, settings: GatewayConfig) -> None:
        del key, response, settings
        raise redis.RedisError("l1 unavailable")


class _MissingL2(L2Cache):
    def __init__(self) -> None:
        self.stores = 0

    async def lookup(self, **kwargs: Any) -> list[SemanticHit]:
        del kwargs
        return []

    async def store(self, **kwargs: Any) -> str:
        del kwargs
        self.stores += 1
        return "entry"

    async def mark_hit(self, entry_id: str) -> None:
        del entry_id


class _CandidateL2(_MissingL2):
    def __init__(self, hits: list[SemanticHit]) -> None:
        self._hits = hits
        self.marked: list[str] = []
        self.deleted: list[str] = []
        self.stores = 0

    async def lookup(self, **kwargs: Any) -> list[SemanticHit]:
        del kwargs
        return self._hits

    async def mark_hit(self, entry_id: str) -> None:
        self.marked.append(entry_id)

    async def delete(self, entry_id: str) -> None:
        self.deleted.append(entry_id)


class _FailingCapacityL2(L2Cache):
    def __init__(self) -> None:
        return None

    async def enforce_capacity(self, capacity: int) -> None:
        del capacity
        raise redis.RedisError("temporary redis failure")


class _SlowEmbedding:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        del texts
        await asyncio.sleep(0.01)
        return [[0.0] * 512]

    async def rerank(self, pairs: list[tuple[str, str]]) -> list[float]:
        del pairs
        await asyncio.sleep(0.01)
        return []


class _RerankEmbedding:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 512 for _ in texts]

    async def rerank(self, pairs: list[tuple[str, str]]) -> list[float]:
        assert len(pairs) == len(self._scores)
        return self._scores


def _req() -> UnifiedRequest:
    return UnifiedRequest(
        request_id="req",
        tenant_id="demo",
        model="chat-large",
        messages=[ChatMessage(role="user", content="退款流程是什么")],
        stream=False,
        temperature=0.1,
        raw_body={},
    )


def _resp(content: str = "ok") -> UnifiedResponse:
    return UnifiedResponse(
        content=content,
        finish_reason="stop",
        model="mock",
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
