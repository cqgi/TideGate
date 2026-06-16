from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import asyncpg
import redis.asyncio as redis
import structlog
from fastapi import FastAPI
from pydantic import ValidationError

from tidegate.config.holder import ReloadResult
from tidegate.config.loader import load_config
from tidegate.config.models import GatewayConfig, TenantConfig
from tidegate.config.tenant_store import TenantStore
from tidegate.providers.manager import close_later

CFG_VERSION_KEY = "cfg:version"
CFG_EVENTS_CHANNEL = "cfg:events"


async def apply_reload(app: FastAPI, *, version: int | None = None) -> ReloadResult:
    holder = app.state.config_holder
    previous: GatewayConfig = holder.current
    try:
        next_config = load_config(holder.path)
        store: TenantStore | None = getattr(app.state, "tenant_store", None)
        if store is not None:
            db_tenants = await store.load_all()
            if db_tenants is not None:
                next_config = merge_tenants(next_config, db_tenants)
    except (OSError, ValueError, ValidationError, asyncpg.PostgresError) as exc:
        return ReloadResult(ok=False, version=holder.version, error=str(exc))
    holder.replace(next_config, version=version)
    manager = app.state.provider_manager
    old_providers = manager.rebuild_if_needed(previous, holder.current)
    if old_providers:
        app.state.task_registry.create(
            close_later(old_providers, previous.server.provider_pool_drain_s),
            name="tidegate-provider-drain",
        )
    return ReloadResult(ok=True, version=holder.version)


def merge_tenants(base: GatewayConfig, tenants: tuple[TenantConfig, ...]) -> GatewayConfig:
    return base.model_copy(update={"tenants": tenants})


async def publish_reload(redis_client: redis.Redis) -> int:
    version = int(await redis_client.incr(CFG_VERSION_KEY))
    await redis_client.publish(CFG_EVENTS_CHANNEL, str(version))
    return version


async def watch_config_events(app: FastAPI, redis_client: redis.Redis) -> None:
    await _supervise_config_loop(
        app,
        loop_name="config_events",
        run_once=lambda: _watch_config_events_once(app, redis_client),
    )


async def _watch_config_events_once(app: FastAPI, redis_client: redis.Redis) -> None:
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(CFG_EVENTS_CHANNEL)
    try:
        async for message in pubsub.listen():
            if message.get("type") != "message":
                continue
            data = message.get("data")
            version = int(data.decode() if isinstance(data, bytes) else data)
            if version != app.state.config_holder.version:
                result = await apply_reload(app, version=version)
                if not result.ok:
                    structlog.get_logger().warning("config_reload_failed", error=result.error)
    finally:
        await pubsub.unsubscribe(CFG_EVENTS_CHANNEL)
        await pubsub.close()


async def poll_config_version(app: FastAPI, redis_client: redis.Redis) -> None:
    await _supervise_config_loop(
        app,
        loop_name="config_poll",
        run_once=lambda: _poll_config_version_loop(app, redis_client),
    )


async def _poll_config_version_loop(app: FastAPI, redis_client: redis.Redis) -> None:
    while True:
        interval_s = app.state.config_holder.current.server.config_poll_interval_s
        await asyncio.sleep(interval_s)
        raw = await redis_client.get(CFG_VERSION_KEY)
        if raw is None:
            continue
        version = int(raw.decode() if isinstance(raw, bytes) else raw)
        if version != app.state.config_holder.version:
            result = await apply_reload(app, version=version)
            if not result.ok:
                structlog.get_logger().warning("config_reload_failed", error=result.error)


async def _supervise_config_loop(
    app: FastAPI,
    *,
    loop_name: str,
    run_once: Callable[[], Awaitable[None]],
) -> None:
    server = app.state.config_holder.current.server
    backoff_s = server.config_reload_backoff_initial_s
    while True:
        try:
            await run_once()
            backoff_s = app.state.config_holder.current.server.config_reload_backoff_initial_s
        except asyncio.CancelledError:
            raise
        except (redis.RedisError, ConnectionError) as exc:
            server = app.state.config_holder.current.server
            structlog.get_logger().error(
                "config_reload_loop_failed",
                loop=loop_name,
                error=str(exc),
                retry_in_s=backoff_s,
            )
            await asyncio.sleep(backoff_s)
            backoff_s = min(server.config_reload_backoff_max_s, backoff_s + backoff_s)
