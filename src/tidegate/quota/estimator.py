from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Protocol

import redis.asyncio as redis
import tiktoken
from redis.exceptions import RedisError

from tidegate.config.models import DeploymentConfig, GatewayConfig
from tidegate.core.models import UnifiedRequest, Usage
from tidegate.quota.keys import correction_key

_LONG_PROMPT_CHARS = 4000
_RedisScalar = bytes | bytearray | memoryview | str | int | float


class CorrectionStore(Protocol):
    def hget(self, key: str, field: str) -> Awaitable[object | None]: ...

    def hset(self, key: str, mapping: Mapping[str, str]) -> Awaitable[object]: ...


class RedisCorrectionStore:
    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client

    async def hget(self, key: str, field: str) -> object | None:
        return await self._redis.hget(key, field)

    async def hset(self, key: str, mapping: Mapping[str, str]) -> object:
        redis_mapping: dict[_RedisScalar, _RedisScalar] = {
            str(field): str(value) for field, value in mapping.items()
        }
        return await self._redis.hset(key, mapping=redis_mapping)


@dataclass(frozen=True)
class Estimate:
    prompt_tokens: int
    output_tokens: int
    tpm_cost: int
    budget_cost_micro: int


class QuotaEstimator:
    def __init__(
        self,
        redis_client: CorrectionStore,
        *,
        process_pool: ProcessPoolExecutor | None = None,
    ) -> None:
        self._redis = redis_client
        self._process_pool = process_pool

    async def estimate(
        self,
        req: UnifiedRequest,
        deployment: DeploymentConfig,
        snapshot: GatewayConfig,
    ) -> Estimate:
        prompt_text = _prompt_text(req)
        prompt_tokens = await _count_tokens(prompt_text, self._process_pool)
        output_est = await self._output_estimate(req, snapshot)
        ratio = await self._ratio(req)
        tpm_cost = max(1, math.ceil((prompt_tokens + output_est) * ratio))
        budget_cost_micro = _budget_micro(prompt_tokens, output_est, deployment)
        return Estimate(
            prompt_tokens=prompt_tokens,
            output_tokens=output_est,
            tpm_cost=tpm_cost,
            budget_cost_micro=budget_cost_micro,
        )

    async def update_correction(
        self,
        *,
        tenant_id: str,
        model: str,
        estimate: Estimate,
        actual: Usage,
        snapshot: GatewayConfig,
    ) -> None:
        key = correction_key(tenant_id, model)
        current_ratio = await self._ratio_for_key(key)
        actual_total = max(1, actual.total_tokens)
        estimated_total = max(1, estimate.tpm_cost)
        sample_ratio = actual_total / estimated_total
        alpha = snapshot.quota_estimator.correction_ewma_alpha
        next_ratio = alpha * sample_ratio + (1 - alpha) * current_ratio
        current_output = await _float_from_redis(await self._redis.hget(key, "output_ewma"), 0.0)
        next_output = alpha * actual.completion_tokens + (1 - alpha) * current_output
        await self._redis.hset(
            key,
            mapping={
                "ewma_ratio": str(next_ratio),
                # EWMA is a compact online approximation for the local P95 estimate.
                "output_ewma": str(next_output),
            },
        )

    async def _output_estimate(self, req: UnifiedRequest, snapshot: GatewayConfig) -> int:
        key = correction_key(req.tenant_id, req.model)
        fallback = snapshot.quota_estimator.output_p95_fallback
        learned = await _float_from_redis(await self._safe_hget(key, "output_ewma"), fallback)
        bounded = math.ceil(learned)
        if req.max_tokens is not None:
            bounded = min(req.max_tokens, bounded)
        return max(1, bounded)

    async def _ratio(self, req: UnifiedRequest) -> float:
        return await self._ratio_for_key(correction_key(req.tenant_id, req.model))

    async def _ratio_for_key(self, key: str) -> float:
        return await _float_from_redis(await self._safe_hget(key, "ewma_ratio"), 1.0)

    async def _safe_hget(self, key: str, field: str) -> object | None:
        try:
            return await self._redis.hget(key, field)
        except (RedisError, ConnectionError, TimeoutError):
            return None


def _budget_micro(
    prompt_tokens: int,
    output_tokens: int,
    deployment: DeploymentConfig,
) -> int:
    cost = (
        prompt_tokens / 1000 * deployment.price_per_1k_input_usd
        + output_tokens / 1000 * deployment.price_per_1k_output_usd
    )
    return max(0, math.ceil(cost * 1_000_000))


def _prompt_text(req: UnifiedRequest) -> str:
    parts = [message.content or "" for message in req.messages]
    return "\n".join(parts)


async def _count_tokens(text: str, process_pool: ProcessPoolExecutor | None) -> int:
    if len(text) > _LONG_PROMPT_CHARS:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(process_pool, _count_tokens_sync, text)
    return _count_tokens_sync(text)


def _count_tokens_sync(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return max(1, len(encoding.encode(text)))


async def _float_from_redis(raw: object | None, default: float) -> float:
    if raw is None:
        return default
    if isinstance(raw, bytes):
        raw = raw.decode()
    if isinstance(raw, (int, float, str)):
        try:
            return float(raw)
        except ValueError:
            return default
    try:
        return float(str(raw))
    except ValueError:
        return default
