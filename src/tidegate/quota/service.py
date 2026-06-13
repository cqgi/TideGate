from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import redis.asyncio as redis
import structlog

from tidegate.config.models import DeploymentConfig, GatewayConfig, TenantConfig
from tidegate.core.errors import ErrorCategory, GatewayError
from tidegate.core.models import UnifiedRequest, Usage
from tidegate.obs.metrics import Metrics
from tidegate.quota.estimator import Estimate, QuotaEstimator
from tidegate.quota.keys import (
    budget_key,
    conc_key,
    reservation_data_key,
    reservation_zset_key,
    rpm_key,
    tpm_key,
)
from tidegate.quota.local_fallback import LocalFallbackLimiter
from tidegate.quota.scripts import QuotaScripts


@dataclass(frozen=True)
class QuotaRejection:
    dim: str
    retry_after_s: float


@dataclass(frozen=True)
class QuotaReservation:
    tenant_id: str
    request_id: str
    model: str
    deployment: DeploymentConfig
    estimate: Estimate
    snapshot: GatewayConfig
    month: str
    redis_reserved: bool = True

    def with_deployment(self, deployment: DeploymentConfig) -> QuotaReservation:
        return QuotaReservation(
            tenant_id=self.tenant_id,
            request_id=self.request_id,
            model=self.model,
            deployment=deployment,
            estimate=self.estimate,
            snapshot=self.snapshot,
            month=self.month,
            redis_reserved=self.redis_reserved,
        )


class QuotaService:
    def __init__(
        self,
        redis_client: redis.Redis,
        scripts: QuotaScripts,
        estimator: QuotaEstimator,
        fallback: LocalFallbackLimiter | None = None,
        metrics: Metrics | None = None,
    ) -> None:
        self._redis = redis_client
        self._scripts = scripts
        self._estimator = estimator
        self._fallback = fallback or LocalFallbackLimiter()
        self._metrics = metrics
        self._fallback_active = False

    async def reserve(
        self,
        *,
        tenant: TenantConfig,
        req: UnifiedRequest,
        deployment: DeploymentConfig,
        snapshot: GatewayConfig,
    ) -> QuotaReservation:
        estimate = await self._estimator.estimate(req, deployment, snapshot)
        plan = snapshot.quota_plans[tenant.plan]
        month = _month()
        now_ms = _now_ms()
        deadline_ms = now_ms + int(snapshot.sweeper.reservation_ttl_s * 1000)
        keys = [
            rpm_key(tenant.id),
            tpm_key(tenant.id),
            conc_key(tenant.id),
            budget_key(tenant.id, month),
            reservation_zset_key(tenant.id),
            reservation_data_key(tenant.id),
        ]
        args = [
            str(now_ms),
            req.request_id,
            str(deadline_ms),
            str(plan.rpm / 60),
            str(plan.rpm),
            str(plan.tpm / 60),
            str(plan.tpm),
            str(estimate.tpm_cost),
            str(plan.concurrent_streams),
            str(estimate.budget_cost_micro),
            str(math.ceil(plan.monthly_budget_usd * 1_000_000)),
            month,
        ]
        try:
            async with asyncio.timeout(0.1):
                result = await self._scripts.check_and_reserve(keys, args)
        except (redis.RedisError, TimeoutError, ConnectionError) as exc:
            return self._reserve_fallback(tenant, req, deployment, estimate, snapshot, exc)
        self._note_redis_recovered()
        if not result or int(_decode_float(result[0])) != 1:
            dim = _decode_dim(result[1])
            retry_after_s = max(1.0, math.ceil(_decode_float(result[2]) / 1000))
            raise GatewayError(
                "quota exceeded",
                ErrorCategory.QUOTA_EXCEEDED,
                retry_after_s=retry_after_s,
                code=f"{dim}_exceeded",
            )
        return QuotaReservation(
            tenant_id=tenant.id,
            request_id=req.request_id,
            model=req.model,
            deployment=deployment,
            estimate=estimate,
            snapshot=snapshot,
            month=month,
        )

    async def settle(
        self,
        reservation: QuotaReservation,
        actual: Usage | None,
        forwarded_tokens: int,
    ) -> None:
        if not reservation.redis_reserved:
            return
        actual_usage = actual or _fallback_usage(reservation.estimate, forwarded_tokens)
        keys = [
            tpm_key(reservation.tenant_id),
            conc_key(reservation.tenant_id),
            budget_key(reservation.tenant_id, reservation.month),
            reservation_zset_key(reservation.tenant_id),
            reservation_data_key(reservation.tenant_id),
        ]
        actual_budget = _actual_budget_micro(actual_usage, reservation.deployment)
        args = [
            reservation.request_id,
            str(actual_usage.total_tokens),
            str(actual_budget),
            reservation.month,
        ]
        try:
            async with asyncio.timeout(reservation.snapshot.sweeper.settle_timeout_s):
                await self._settle_with_retry(reservation, keys, args)
        except (redis.RedisError, TimeoutError, ConnectionError) as exc:
            self._record_settle_failure(reservation, exc)
            return
        try:
            await self._estimator.update_correction(
                tenant_id=reservation.tenant_id,
                model=reservation.model,
                estimate=reservation.estimate,
                actual=actual_usage,
                snapshot=reservation.snapshot,
            )
        except redis.RedisError as exc:
            structlog.get_logger().warning(
                "quota_correction_update_failed",
                tenant=reservation.tenant_id,
                error=str(exc),
            )

    async def refund_full(self, reservation: QuotaReservation) -> None:
        zero = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        await self.settle(reservation, zero, 0)

    async def sweep_tenant(self, snapshot: GatewayConfig, tenant: TenantConfig) -> int:
        month = _month()
        keys = [
            tpm_key(tenant.id),
            conc_key(tenant.id),
            budget_key(tenant.id, month),
            reservation_zset_key(tenant.id),
            reservation_data_key(tenant.id),
        ]
        result = await self._scripts.sweep(
            keys,
            [str(_now_ms()), str(snapshot.sweeper.batch_limit), month],
        )
        return int(_decode_float(result[0]))

    async def debug_state(self, tenant_id: str) -> dict[str, object]:
        month = _month()
        tpm_raw = await self._redis.hget(tpm_key(tenant_id), "tokens")
        rpm_raw = await self._redis.hget(rpm_key(tenant_id), "tokens")
        conc_raw = await self._redis.get(conc_key(tenant_id))
        budget_raw = await self._redis.get(budget_key(tenant_id, month))
        resv_count = await self._redis.zcard(reservation_zset_key(tenant_id))
        return {
            "rpm_tokens": _decode_float(rpm_raw),
            "tpm_tokens": _decode_float(tpm_raw),
            "conc": int(_decode_float(conc_raw)),
            "budget": int(_decode_float(budget_raw)),
            "resv": int(resv_count),
        }

    def _reserve_fallback(
        self,
        tenant: TenantConfig,
        req: UnifiedRequest,
        deployment: DeploymentConfig,
        estimate: Estimate,
        snapshot: GatewayConfig,
        exc: BaseException,
    ) -> QuotaReservation:
        plan = snapshot.quota_plans[tenant.plan]
        if plan.fail_mode != "open":
            raise GatewayError(
                "quota backend unavailable",
                ErrorCategory.INTERNAL,
                http_status=503,
            ) from exc
        if not self._fallback_active:
            self._fallback_active = True
            structlog.get_logger().error("quota_fallback_entered", tenant=tenant.id, error=str(exc))
        if not self._fallback.allow(tenant.id, plan, estimate):
            raise GatewayError(
                "local quota exceeded",
                ErrorCategory.QUOTA_EXCEEDED,
                retry_after_s=1.0,
                code="tpm_exceeded",
            ) from exc
        # Open-mode tenants fall back to a local limiter without writing a Redis
        # reservation; later reconciliation absorbs drift when Redis recovers.
        return QuotaReservation(
            tenant_id=tenant.id,
            request_id=req.request_id,
            model=req.model,
            deployment=deployment,
            estimate=estimate,
            snapshot=snapshot,
            month=_month(),
            redis_reserved=False,
        )

    def _note_redis_recovered(self) -> None:
        if self._fallback_active:
            self._fallback_active = False
            structlog.get_logger().info("quota_fallback_exited")

    async def _settle_with_retry(
        self,
        reservation: QuotaReservation,
        keys: list[str],
        args: list[str],
    ) -> None:
        try:
            await self._scripts.settle(keys, args)
        except redis.RedisError as exc:
            structlog.get_logger().warning(
                "quota_settle_failed_once",
                tenant=reservation.tenant_id,
                error=str(exc),
            )
            await self._scripts.settle(keys, args)

    def _record_settle_failure(self, reservation: QuotaReservation, exc: BaseException) -> None:
        structlog.get_logger().error(
            "quota_settle_failed",
            tenant=reservation.tenant_id,
            error=str(exc),
        )
        if self._metrics is not None:
            self._metrics.quota_settle_failed.labels(reservation.tenant_id).inc()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _month() -> str:
    return datetime.now(UTC).strftime("%Y-%m")


def _decode_dim(raw: object) -> str:
    if isinstance(raw, bytes):
        raw = raw.decode()
    dim = str(raw)
    if dim == "concurrency":
        return "concurrency"
    return dim


def _decode_float(raw: object | None) -> float:
    if raw is None:
        return 0.0
    if isinstance(raw, bytes):
        raw = raw.decode()
    if isinstance(raw, (int, float, str)):
        return float(raw)
    return float(str(raw))


def _fallback_usage(estimate: Estimate, forwarded_tokens: int) -> Usage:
    completion = max(0, forwarded_tokens)
    prompt = max(0, estimate.tpm_cost - estimate.output_tokens)
    # Missing upstream usage is settled with forwarded tokens plus the reserved prompt
    # estimate so disconnects still release concurrency exactly once.
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


def _actual_budget_micro(actual: Usage, deployment: DeploymentConfig) -> int:
    cost = (
        actual.prompt_tokens / 1000 * deployment.price_per_1k_input_usd
        + actual.completion_tokens / 1000 * deployment.price_per_1k_output_usd
    )
    return max(0, math.ceil(cost * 1_000_000))
