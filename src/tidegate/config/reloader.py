from __future__ import annotations

import asyncio

import redis.asyncio as redis
import structlog
from fastapi import FastAPI

from tidegate.config.holder import ReloadResult
from tidegate.config.models import GatewayConfig
from tidegate.providers.manager import close_later

CFG_VERSION_KEY = "cfg:version"
CFG_EVENTS_CHANNEL = "cfg:events"


async def apply_reload(app: FastAPI, *, version: int | None = None) -> ReloadResult:
    holder = app.state.config_holder
    previous: GatewayConfig = holder.current
    result: ReloadResult = holder.reload(version)
    if not result.ok:
        return result
    manager = app.state.provider_manager
    old_providers = manager.rebuild_if_needed(previous, holder.current)
    if old_providers:
        app.state.task_registry.create(
            close_later(old_providers, previous.server.provider_pool_drain_s),
            name="tidegate-provider-drain",
        )
    return result


async def publish_reload(redis_client: redis.Redis) -> int:
    version = int(await redis_client.incr(CFG_VERSION_KEY))
    await redis_client.publish(CFG_EVENTS_CHANNEL, str(version))
    return version


async def watch_config_events(app: FastAPI, redis_client: redis.Redis) -> None:
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
