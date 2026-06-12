from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import redis.asyncio as redis
import structlog
from fastapi import FastAPI

from tidegate.api.admin import router as admin_router
from tidegate.api.errors import install_exception_handlers
from tidegate.api.middleware import AuthMiddleware, RequestContextMiddleware
from tidegate.api.routes import router
from tidegate.config.holder import ConfigHolder
from tidegate.config.models import GatewayConfig
from tidegate.config.reloader import poll_config_version, watch_config_events
from tidegate.obs.logging import configure_logging
from tidegate.obs.loop_lag import probe_loop_lag
from tidegate.obs.metrics import Metrics
from tidegate.providers.manager import ProviderManager
from tidegate.quota.estimator import QuotaEstimator, RedisCorrectionStore
from tidegate.quota.local_fallback import LocalFallbackLimiter
from tidegate.quota.scripts import QuotaScripts
from tidegate.quota.service import QuotaService
from tidegate.quota.sweeper import sweep_loop


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


def create_app(settings: GatewayConfig, config_path: str | Path = "config/gateway.yaml") -> FastAPI:
    configure_logging()
    metrics = Metrics.create()
    holder = ConfigHolder(settings, Path(config_path))
    provider_manager = ProviderManager(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        task_registry = TaskRegistry()
        redis_client = redis.from_url(settings.redis.url, decode_responses=False)
        cpu_pool = ProcessPoolExecutor(max_workers=settings.server.cpu_pool_workers)
        quota_scripts = QuotaScripts(redis_client)
        quota_service = QuotaService(
            redis_client,
            quota_scripts,
            QuotaEstimator(RedisCorrectionStore(redis_client), process_pool=cpu_pool),
            LocalFallbackLimiter(),
            metrics,
        )
        app.state.config_holder = holder
        app.state.metrics = metrics
        app.state.provider_manager = provider_manager
        app.state.task_registry = task_registry
        app.state.redis = redis_client
        app.state.quota = quota_service
        app.state.cpu_pool = cpu_pool
        task_registry.create(
            probe_loop_lag(metrics, settings.server.loop_lag_interval_s),
            name="tidegate-loop-lag",
        )
        try:
            await redis_client.ping()
            await quota_scripts.load()
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
        try:
            yield
        finally:
            await task_registry.drain()
            await provider_manager.close()
            await redis_client.aclose()
            cpu_pool.shutdown(cancel_futures=True)

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
