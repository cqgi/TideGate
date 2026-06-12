from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import asyncpg
import redis.asyncio as redis
import structlog
from fastapi import FastAPI

from tidegate.api.admin import router as admin_router
from tidegate.api.errors import install_exception_handlers
from tidegate.api.middleware import AuthMiddleware, RequestContextMiddleware
from tidegate.api.routes import router
from tidegate.cache.embedding import EmbeddingService, init_embedding_worker
from tidegate.cache.l1 import L1Cache
from tidegate.cache.l2 import L2Cache, capacity_sweep_loop
from tidegate.cache.service import CacheService
from tidegate.config.holder import ConfigHolder
from tidegate.config.models import GatewayConfig
from tidegate.config.reloader import poll_config_version, watch_config_events
from tidegate.obs.logging import configure_logging
from tidegate.obs.loop_lag import probe_loop_lag
from tidegate.obs.metrics import Metrics
from tidegate.obs.otel import configure_otel
from tidegate.providers.manager import ProviderManager
from tidegate.quota.estimator import QuotaEstimator, RedisCorrectionStore
from tidegate.quota.local_fallback import LocalFallbackLimiter
from tidegate.quota.scripts import QuotaScripts
from tidegate.quota.service import QuotaService
from tidegate.quota.sweeper import sweep_loop
from tidegate.routing.hedge import HedgeBudget
from tidegate.routing.reporter import prewarm_from_aggregate, report_loop
from tidegate.routing.selector import P2CSelector
from tidegate.routing.stats import RoutingState
from tidegate.settlement import LedgerBatcher


@dataclass
class TaskRegistry:
    tasks: set[asyncio.Task[Any]] = field(default_factory=set)

    def create(self, coro: Any, *, name: str) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro, name=name)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    async def drain(self) -> None:
        for task in tuple(self.tasks):
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)


class ActiveStreamTracker:
    def __init__(self) -> None:
        self._active = 0
        self._done = asyncio.Event()
        self._done.set()

    def enter(self) -> None:
        self._active += 1
        self._done.clear()

    def exit(self) -> None:
        self._active -= 1
        if self._active <= 0:
            self._active = 0
            self._done.set()

    async def drain(self, timeout_s: float) -> None:
        try:
            async with asyncio.timeout(timeout_s):
                await self._done.wait()
        except TimeoutError:
            structlog.get_logger().warning(
                "active_stream_drain_timeout",
                active=self._active,
            )


def create_app(settings: GatewayConfig, config_path: str | Path = "config/gateway.yaml") -> FastAPI:
    configure_logging()
    configure_otel(settings.otel)
    metrics = Metrics.create()
    holder = ConfigHolder(settings, Path(config_path))
    provider_manager = ProviderManager(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        task_registry = TaskRegistry()
        redis_client = redis.from_url(settings.redis.url, decode_responses=False)
        # REWORK-M2-5: token counting uses this shared CPU pool; M4 embedding has its
        # own pool below because fastembed model weights live in worker initializers.
        cpu_pool = ProcessPoolExecutor(max_workers=settings.server.cpu_pool_workers)
        embedding_pool: ProcessPoolExecutor | None = None
        embedding_service: EmbeddingService | None = None
        pg_pool: asyncpg.Pool | None = None
        if any(tenant.cache.l2 for tenant in settings.tenants):
            embedding_pool = ProcessPoolExecutor(
                max_workers=settings.cache.l2.embed_pool_workers,
                initializer=init_embedding_worker,
                initargs=(
                    settings.cache.l2.embedding_model,
                    settings.cache.l2.model_cache_dir,
                    settings.cache.l2.hf_endpoint,
                    settings.cache.l2.reranker_model,
                ),
            )
            embedding_service = EmbeddingService(embedding_pool)
        quota_scripts = QuotaScripts(redis_client)
        routing_state = RoutingState(settings, metrics)
        selector = P2CSelector(settings, routing_state)
        l2_cache = L2Cache(redis_client)
        cache_service = CacheService(
            L1Cache(redis_client),
            l2_cache,
            embedding_service,
            metrics,
        )
        quota_service = QuotaService(
            redis_client,
            quota_scripts,
            QuotaEstimator(RedisCorrectionStore(redis_client), process_pool=cpu_pool),
            LocalFallbackLimiter(),
            metrics,
        )
        dsn = os.environ.get(settings.postgres.dsn_env)
        if dsn:
            try:
                pg_pool = await asyncpg.create_pool(dsn)
                ddl = await asyncio.to_thread(_ledger_schema_sql)
                async with pg_pool.acquire() as conn:
                    await conn.execute(ddl)
            except (OSError, asyncpg.PostgresError) as exc:
                structlog.get_logger().warning(
                    "postgres_unavailable_ledger_disabled",
                    error=str(exc),
                )
                if pg_pool is not None:
                    await pg_pool.close()
                pg_pool = None
        ledger = LedgerBatcher(pg_pool, settings.settlement, metrics)
        app.state.config_holder = holder
        app.state.metrics = metrics
        app.state.provider_manager = provider_manager
        app.state.task_registry = task_registry
        app.state.active_streams = ActiveStreamTracker()
        app.state.redis = redis_client
        app.state.quota = quota_service
        app.state.cpu_pool = cpu_pool
        app.state.embedding_pool = embedding_pool
        app.state.cache = cache_service
        app.state.ledger = ledger
        app.state.routing_state = routing_state
        app.state.selector = selector
        app.state.hedge_budget = HedgeBudget()
        task_registry.create(
            probe_loop_lag(metrics, settings.server.loop_lag_interval_s),
            name="tidegate-loop-lag",
        )
        try:
            await redis_client.ping()
            await quota_scripts.load()
            await l2_cache.ensure_index()
            await prewarm_from_aggregate(
                redis_client,
                settings,
                routing_state,
                now_s=asyncio.get_running_loop().time(),
            )
        except redis.RedisError as exc:
            # DECISION: M2 keeps serving so per-tenant fail_mode can decide open/closed fallback.
            structlog.get_logger().warning("redis_unavailable_hot_reload_disabled", error=str(exc))
        else:
            task_registry.create(
                watch_config_events(app, redis_client), name="tidegate-config-events"
            )
            task_registry.create(
                poll_config_version(app, redis_client), name="tidegate-config-poll"
            )
            task_registry.create(
                sweep_loop(holder, quota_service, metrics), name="tidegate-quota-sweeper"
            )
            task_registry.create(
                report_loop(redis_client, settings, routing_state), name="tidegate-routing-report"
            )
            if embedding_service is not None:
                task_registry.create(
                    capacity_sweep_loop(l2_cache, lambda: holder.current),
                    name="tidegate-cache-l2-capacity-sweep",
                )
            if pg_pool is not None:
                task_registry.create(ledger.run(), name="tidegate-ledger-batcher")
        try:
            yield
        finally:
            await app.state.active_streams.drain(settings.settlement.drain_timeout_s)
            await task_registry.drain()
            await ledger.drain()
            ledger.close()
            await provider_manager.close()
            await redis_client.aclose()
            if pg_pool is not None:
                await pg_pool.close()
            cpu_pool.shutdown(cancel_futures=True)
            if embedding_pool is not None:
                embedding_pool.shutdown(cancel_futures=True)

    app = FastAPI(title="TideGate", lifespan=lifespan)
    app.state.config_holder = holder
    app.state.provider_manager = provider_manager
    app.state.metrics = metrics
    app.add_middleware(AuthMiddleware, config=holder)
    app.add_middleware(RequestContextMiddleware)
    install_exception_handlers(app)
    app.include_router(admin_router)
    app.include_router(router)
    return app


def _ledger_schema_sql() -> str:
    return Path("deploy/sql/001_usage_ledger.sql").read_text(encoding="utf-8")
