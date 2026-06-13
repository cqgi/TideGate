from __future__ import annotations

import asyncio
import json
import os
import socket
from typing import Any

import redis.asyncio as redis
import structlog

from tidegate.config.models import DeploymentConfig, GatewayConfig
from tidegate.routing.breaker import BreakerState
from tidegate.routing.stats import RoutingState


def instance_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


async def prewarm_from_aggregate(
    redis_client: redis.Redis,
    settings: GatewayConfig,
    routing_state: RoutingState,
    *,
    now_s: float,
) -> None:
    # Aggregation is dashboard and prewarm data only, never the hot-path source of truth.
    for deployment in _deployments(settings):
        key = _cb_key(deployment)
        try:
            values = await redis_client.hvals(key)
        except redis.RedisError as exc:
            structlog.get_logger().warning("routing_prewarm_failed", error=str(exc))
            return
        states: list[tuple[str, float]] = []
        for value in values:
            decoded = _decode_json(value)
            if not isinstance(decoded, dict):
                continue
            state = decoded.get("state")
            remaining = decoded.get("remaining_s", 0.0)
            if isinstance(state, str) and isinstance(remaining, int | float):
                states.append((state, float(remaining)))
        if not states:
            continue
        open_states = [remaining for state, remaining in states if state == BreakerState.OPEN.value]
        if len(open_states) > len(states) / 2:
            routing_state.force_open(
                deployment,
                now_s=now_s,
                remaining_s=sum(open_states) / len(open_states),
            )


async def report_loop(
    redis_client: redis.Redis,
    settings: GatewayConfig,
    routing_state: RoutingState,
) -> None:
    ident = instance_id()
    while True:
        await asyncio.sleep(settings.routing.agg_report_interval_s)
        try:
            await report_once(redis_client, settings, routing_state, instance=ident)
        except asyncio.CancelledError:
            raise
        except redis.RedisError as exc:
            structlog.get_logger().warning("routing_agg_report_failed", error=str(exc))


async def report_once(
    redis_client: redis.Redis,
    settings: GatewayConfig,
    routing_state: RoutingState,
    *,
    instance: str | None = None,
) -> None:
    ident = instance or instance_id()
    now_s = asyncio.get_running_loop().time()
    pipe = redis_client.pipeline()
    snapshot = routing_state.snapshot()
    for deployment in _deployments(settings):
        name = f"{deployment.provider}/{deployment.upstream_model}"
        data = snapshot.get(name)
        if data is None:
            continue
        state = str(data["state"])
        open_until = data.get("open_until_s", 0.0)
        open_until_s = float(open_until) if isinstance(open_until, int | float) else 0.0
        cb_value = json.dumps(
            {
                "ts": now_s,
                "state": state,
                "remaining_s": max(0.0, open_until_s - now_s),
            },
            separators=(",", ":"),
        )
        stats_value = json.dumps(
            {
                "ts": now_s,
                "ewma_ttft": data["ewma_ttft"],
                "ewma_err": data["ewma_err"],
                "inflight": data["inflight"],
            },
            separators=(",", ":"),
        )
        cb_key = _cb_key(deployment)
        stats_key = _stats_key(deployment)
        pipe.hset(cb_key, ident, cb_value)
        pipe.expire(cb_key, settings.routing.agg_ttl_s)
        pipe.hset(stats_key, ident, stats_value)
        pipe.expire(stats_key, settings.routing.agg_ttl_s)
    await pipe.execute()


def _deployments(settings: GatewayConfig) -> list[DeploymentConfig]:
    seen: set[tuple[str, str]] = set()
    deployments: list[DeploymentConfig] = []
    for group in settings.model_groups.values():
        for deployment in group.deployments:
            key = (deployment.provider, deployment.upstream_model)
            if key in seen:
                continue
            seen.add(key)
            deployments.append(deployment)
    return deployments


def _decode_json(value: Any) -> object:
    if isinstance(value, bytes):
        value = value.decode()
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _cb_key(deployment: DeploymentConfig) -> str:
    return f"agg:cb:{deployment.provider}:{deployment.upstream_model}"


def _stats_key(deployment: DeploymentConfig) -> str:
    return f"agg:stats:{deployment.provider}:{deployment.upstream_model}"
