from __future__ import annotations

import asyncio
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import redis.asyncio as redis
import structlog
import ulid

from tidegate.cache.keys import semcache_key
from tidegate.config.models import GatewayConfig

INDEX_NAME = "idx:semcache"
VECTOR_DIM = 512


@dataclass(frozen=True)
class SemanticHit:
    entry_id: str
    l1_key: str
    score: float


class L2Cache:
    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client

    async def ensure_index(self) -> None:
        try:
            await self._execute("FT.INFO", INDEX_NAME)
            return
        except redis.ResponseError as exc:
            if "unknown index name" not in str(exc).lower():
                raise
        await self._execute(
            "FT.CREATE",
            INDEX_NAME,
            "ON",
            "HASH",
            "PREFIX",
            "1",
            "semcache:",
            "SCHEMA",
            "tenant",
            "TAG",
            "prompt_version",
            "TAG",
            "vec",
            "VECTOR",
            "HNSW",
            "6",
            "TYPE",
            "FLOAT32",
            "DIM",
            str(VECTOR_DIM),
            "DISTANCE_METRIC",
            "COSINE",
        )

    async def lookup(
        self,
        *,
        tenant_id: str,
        prompt_version: str,
        vector: list[float],
        threshold: float,
        timeout_ms: int,
    ) -> SemanticHit | None:
        try:
            async with asyncio.timeout(timeout_ms / 1000):
                query = (
                    f"(@tenant:{{{_tag_value(tenant_id)}}} "
                    f"@prompt_version:{{{_tag_value(prompt_version)}}})"
                    "=>[KNN 1 @vec $vec AS dist]"
                )
                rows = await self._execute(
                    "FT.SEARCH",
                    INDEX_NAME,
                    query,
                    "PARAMS",
                    "2",
                    "vec",
                    _pack(vector),
                    "RETURN",
                    "2",
                    "l1_key",
                    "dist",
                    "SORTBY",
                    "dist",
                    "DIALECT",
                    "2",
                )
        except (TimeoutError, redis.RedisError):
            return None
        parsed = _parse_search(rows)
        if parsed is None:
            return None
        key, l1_key, distance = parsed
        score = 1.0 - distance
        if score < threshold:
            return None
        entry_id = key.removeprefix("semcache:")
        await self._redis.hset(key, mapping={"last_hit_at": str(time.time())})
        return SemanticHit(entry_id=entry_id, l1_key=l1_key, score=score)

    async def store(
        self,
        *,
        tenant_id: str,
        prompt_version: str,
        vector: list[float],
        l1_key: str,
        settings: GatewayConfig,
    ) -> str:
        entry_id = str(ulid.new())
        now = str(time.time())
        await self._redis.hset(
            semcache_key(entry_id),
            mapping={
                "tenant": tenant_id,
                "prompt_version": prompt_version,
                "vec": _pack(vector),
                "l1_key": l1_key,
                "created_at": now,
                "last_hit_at": now,
            },
        )
        return entry_id

    async def delete(self, entry_id: str) -> None:
        await self._redis.delete(semcache_key(entry_id))

    async def _execute(self, *args: object) -> object:
        return cast(object, await self._redis.execute_command(*args))  # type: ignore[no-untyped-call]

    async def enforce_capacity(self, capacity: int) -> None:
        # DECISION: M4 uses a bounded best-effort SCAN cleanup; exact global ordering is
        # unnecessary for the small personal-project Redis index size.
        entries: list[tuple[float, str]] = []
        async for key in self._redis.scan_iter(match="semcache:*"):
            raw = await self._redis.hget(key, "last_hit_at")
            name = key.decode() if isinstance(key, bytes) else str(key)
            entries.append((_float(raw), name))
        if len(entries) <= capacity:
            return
        for _, key in sorted(entries)[: max(0, len(entries) - capacity)]:
            await self._redis.delete(key)


async def capacity_sweep_loop(
    l2: L2Cache,
    settings_provider: Callable[[], GatewayConfig],
) -> None:
    while True:
        settings = settings_provider()
        await asyncio.sleep(settings.cache.l2.capacity_sweep_interval_s)
        try:
            await l2.enforce_capacity(settings.cache.l2.index_capacity)
        except redis.RedisError as exc:
            structlog.get_logger().warning("cache_l2_capacity_sweep_failed", error=str(exc))


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _tag_value(value: str) -> str:
    replacements: dict[str | int, str | int | None] = {char: f"\\{char}" for char in " ,{}|"}
    return value.translate(str.maketrans(replacements))


def _parse_search(rows: Any) -> tuple[str, str, float] | None:
    if isinstance(rows, dict):
        return _parse_search_dict(rows)
    if not isinstance(rows, list) or not rows or int(rows[0]) == 0:
        return None
    if len(rows) < 3:
        return None
    key = rows[1]
    fields = rows[2]
    if isinstance(key, bytes):
        key = key.decode()
    if not isinstance(fields, list):
        return None
    values: dict[str, object] = {}
    for index in range(0, len(fields), 2):
        field = fields[index]
        value = fields[index + 1]
        if isinstance(field, bytes):
            field = field.decode()
        values[str(field)] = value
    l1_key = values.get("l1_key")
    distance = values.get("dist")
    if isinstance(l1_key, bytes):
        l1_key = l1_key.decode()
    if isinstance(distance, bytes):
        distance = distance.decode()
    if not isinstance(l1_key, str) or not isinstance(distance, str | float | int):
        return None
    return str(key), l1_key, float(distance)


def _parse_search_dict(rows: dict[Any, Any]) -> tuple[str, str, float] | None:
    total = _dict_get(rows, "total_results")
    if not isinstance(total, int) or total == 0:
        return None
    results = _dict_get(rows, "results")
    if not isinstance(results, list) or not results:
        return None
    row = results[0]
    if not isinstance(row, dict):
        return None
    key = _dict_get(row, "id")
    attrs = _dict_get(row, "extra_attributes")
    if isinstance(key, bytes):
        key = key.decode()
    if not isinstance(key, str) or not isinstance(attrs, dict):
        return None
    l1_key = _dict_get(attrs, "l1_key")
    distance = _dict_get(attrs, "dist")
    if isinstance(l1_key, bytes):
        l1_key = l1_key.decode()
    if isinstance(distance, bytes):
        distance = distance.decode()
    if not isinstance(l1_key, str) or not isinstance(distance, str | float | int):
        return None
    return key, l1_key, float(distance)


def _dict_get(data: dict[Any, Any], name: str) -> Any:
    if name in data:
        return data[name]
    return data.get(name.encode())


def _float(value: object | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str | int | float):
        return float(value)
    return 0.0
