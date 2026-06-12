from __future__ import annotations

import gzip
import json
import random

import redis.asyncio as redis
from pydantic import ValidationError

from tidegate.config.models import GatewayConfig
from tidegate.core.models import UnifiedResponse


class L1Cache:
    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client

    async def get(self, key: str) -> UnifiedResponse | None:
        raw = await self._redis.get(key)
        if raw is None:
            return None
        raw_bytes = raw.encode("latin1") if isinstance(raw, str) else bytes(raw)
        try:
            payload = json.loads(gzip.decompress(raw_bytes).decode("utf-8"))
            return UnifiedResponse.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValidationError):
            await self._redis.delete(key)
            return None

    async def set(self, key: str, response: UnifiedResponse, settings: GatewayConfig) -> None:
        raw = response.model_dump_json().encode("utf-8")
        ttl = _ttl(settings)
        await self._redis.set(key, gzip.compress(raw), ex=ttl)

    async def delete(self, key: str) -> None:
        await self._redis.delete(key)


def _ttl(settings: GatewayConfig) -> int:
    ttl_s = settings.cache.l1.ttl_s
    jitter = settings.cache.l1.ttl_jitter_ratio
    if jitter <= 0:
        return ttl_s
    factor = 1 + random.uniform(-jitter, jitter)
    return max(1, round(ttl_s * factor))
