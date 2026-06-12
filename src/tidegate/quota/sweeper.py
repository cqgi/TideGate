from __future__ import annotations

import asyncio

import redis.asyncio as redis
import structlog

from tidegate.config.holder import ConfigHolder
from tidegate.obs.metrics import Metrics
from tidegate.quota.service import QuotaService


async def sweep_loop(holder: ConfigHolder, quota: QuotaService, metrics: Metrics) -> None:
    while True:
        snapshot = holder.current
        await asyncio.sleep(snapshot.sweeper.interval_s)
        for tenant in snapshot.tenants:
            try:
                swept = await quota.sweep_tenant(snapshot, tenant)
            except redis.RedisError as exc:
                structlog.get_logger().warning(
                    "quota_sweep_failed",
                    tenant=tenant.id,
                    error=str(exc),
                )
                continue
            if swept > 0:
                metrics.quota_sweep.labels(tenant.id).inc(swept)
                structlog.get_logger().warning(
                    "quota_reservations_swept",
                    tenant=tenant.id,
                    count=swept,
                )
