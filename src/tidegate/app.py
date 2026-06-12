from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI

from tidegate.api.errors import install_exception_handlers
from tidegate.api.middleware import AuthMiddleware, RequestContextMiddleware
from tidegate.api.routes import router
from tidegate.config.models import ConfigHolder, GatewayConfig
from tidegate.obs.logging import configure_logging
from tidegate.obs.loop_lag import probe_loop_lag
from tidegate.obs.metrics import Metrics
from tidegate.providers.registry import build_providers


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


def create_app(settings: GatewayConfig) -> FastAPI:
    configure_logging()
    metrics = Metrics.create()
    holder = ConfigHolder(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        limits = httpx.Limits(max_connections=_max_connections(settings))
        client = httpx.AsyncClient(limits=limits, trust_env=False)
        task_registry = TaskRegistry()
        app.state.config_holder = holder
        app.state.http_client = client
        app.state.metrics = metrics
        app.state.providers = build_providers(settings, client)
        app.state.task_registry = task_registry
        task_registry.create(
            probe_loop_lag(metrics, settings.server.loop_lag_interval_s),
            name="tidegate-loop-lag",
        )
        try:
            yield
        finally:
            await task_registry.drain()
            await client.aclose()

    app = FastAPI(title="TideGate", lifespan=lifespan)
    app.state.config_holder = holder
    app.state.metrics = metrics
    app.add_middleware(AuthMiddleware, config=holder)
    app.add_middleware(RequestContextMiddleware)
    install_exception_handlers(app)
    app.include_router(router)
    return app


def _max_connections(settings: GatewayConfig) -> int:
    configured = [provider.max_connections for provider in settings.providers.values()]
    if not configured:
        return 1
    return max(configured)
