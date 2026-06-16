from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import redis.asyncio as redis

from tidegate.config.holder import ConfigHolder
from tidegate.config.loader import load_config
from tidegate.config.models import CacheToggleConfig, TenantConfig
from tidegate.config.reloader import (
    CFG_EVENTS_CHANNEL,
    CFG_VERSION_KEY,
    _watch_config_events_once,
    apply_reload,
    merge_tenants,
    poll_config_version,
    publish_reload,
)


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.published: list[tuple[str, str]] = []

    async def incr(self, key: str) -> int:
        current = int(self.values.get(key, b"0").decode())
        current += 1
        self.values[key] = str(current).encode()
        return current

    async def publish(self, channel: str, value: str) -> None:
        self.published.append((channel, value))

    async def get(self, key: str) -> bytes | None:
        return self.values.get(key)


class FlakyRedis(FakeRedis):
    def __init__(self) -> None:
        super().__init__()
        self.get_calls = 0

    async def get(self, key: str) -> bytes | None:
        self.get_calls += 1
        if self.get_calls == 1:
            raise redis.ConnectionError("temporary outage")
        return await super().get(key)


class EventRedis(FakeRedis):
    def __init__(self, messages: list[dict[str, object]]) -> None:
        super().__init__()
        self.pubsub_client = FakePubSub(messages)

    def pubsub(self) -> FakePubSub:
        return self.pubsub_client


class FakePubSub:
    def __init__(self, messages: list[dict[str, object]]) -> None:
        self.messages = messages
        self.subscribed: list[str] = []
        self.unsubscribed: list[str] = []
        self.closed = False

    async def subscribe(self, channel: str) -> None:
        self.subscribed.append(channel)

    async def listen(self) -> Any:
        for message in self.messages:
            yield message

    async def unsubscribe(self, channel: str) -> None:
        self.unsubscribed.append(channel)

    async def close(self) -> None:
        self.closed = True


class DummyProviderManager:
    def __init__(self) -> None:
        self.previous: object | None = None
        self.current: object | None = None

    def rebuild_if_needed(self, previous: object, current: object) -> list[Any]:
        self.previous = previous
        self.current = current
        return []


class DummyTenantStore:
    def __init__(self, tenants: tuple[TenantConfig, ...] | None) -> None:
        self.tenants = tenants
        self.loads = 0

    async def load_all(self) -> tuple[TenantConfig, ...] | None:
        self.loads += 1
        return self.tenants


class FailingTenantStore:
    async def load_all(self) -> tuple[TenantConfig, ...] | None:
        raise ValueError("db tenant load failed")


@pytest.mark.asyncio
async def test_publish_reload_increments_and_publishes() -> None:
    redis = FakeRedis()
    version = await publish_reload(redis)  # type: ignore[arg-type]
    assert version == 1
    assert redis.values[CFG_VERSION_KEY] == b"1"
    assert redis.published == [(CFG_EVENTS_CHANNEL, "1")]


@pytest.mark.asyncio
async def test_poll_config_version_applies_new_version(poll_config_path: Path) -> None:
    holder = ConfigHolder(load_config(poll_config_path), poll_config_path)
    redis = FakeRedis()
    redis.values[CFG_VERSION_KEY] = b"1"
    app = SimpleNamespace(
        state=SimpleNamespace(
            config_holder=holder,
            provider_manager=DummyProviderManager(),
            task_registry=SimpleNamespace(create=lambda *args, **kwargs: None),
        )
    )

    task = asyncio.create_task(poll_config_version(app, redis))  # type: ignore[arg-type]
    try:
        for _ in range(20):
            if holder.version == 1:
                break
            await asyncio.sleep(0.01)
        assert holder.version == 1
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_merge_tenants_replaces_only_tenant_tuple() -> None:
    base = load_config("tests/fixtures/gateway-test.yaml")
    tenant = TenantConfig(
        id="db-demo",
        api_key_sha256="db-key",
        plan="free",
        policy="default",
        cache=CacheToggleConfig(l1=False, l2=False),
    )

    merged = merge_tenants(base, (tenant,))

    assert merged.tenants == (tenant,)
    assert merged.providers == base.providers
    assert merged.model_groups == base.model_groups
    assert merged is not base


@pytest.mark.asyncio
async def test_apply_reload_merges_db_tenants_after_yaml_reload(poll_config_path: Path) -> None:
    holder = ConfigHolder(load_config(poll_config_path), poll_config_path)
    db_tenant = TenantConfig(
        id="db-demo",
        api_key_sha256="db-key",
        plan="free",
        policy="default",
        cache=CacheToggleConfig(l1=False, l2=False),
    )
    manager = DummyProviderManager()
    store = DummyTenantStore((db_tenant,))
    app = SimpleNamespace(
        state=SimpleNamespace(
            config_holder=holder,
            provider_manager=manager,
            tenant_store=store,
            task_registry=SimpleNamespace(create=lambda *args, **kwargs: None),
        )
    )

    result = await apply_reload(app, version=2)  # type: ignore[arg-type]

    assert result.ok
    assert holder.version == 2
    assert holder.current.tenants == (db_tenant,)
    assert store.loads == 1
    assert manager.previous is not None
    assert manager.current == holder.current


@pytest.mark.asyncio
async def test_apply_reload_keeps_previous_snapshot_when_db_tenant_load_fails(
    poll_config_path: Path,
) -> None:
    holder = ConfigHolder(load_config(poll_config_path), poll_config_path)
    previous = holder.current
    app = SimpleNamespace(
        state=SimpleNamespace(
            config_holder=holder,
            provider_manager=DummyProviderManager(),
            tenant_store=FailingTenantStore(),
            task_registry=SimpleNamespace(create=lambda *args, **kwargs: None),
        )
    )

    result = await apply_reload(app, version=3)  # type: ignore[arg-type]

    assert not result.ok
    assert result.version == 0
    assert "db tenant load failed" in str(result.error)
    assert holder.current is previous
    assert holder.version == 0


@pytest.mark.asyncio
async def test_config_event_reloads_db_tenants_through_existing_channel(
    poll_config_path: Path,
) -> None:
    holder = ConfigHolder(load_config(poll_config_path), poll_config_path)
    db_tenant = TenantConfig(
        id="db-demo",
        api_key_sha256="db-key",
        plan="free",
        policy="default",
        cache=CacheToggleConfig(l1=False, l2=False),
    )
    manager = DummyProviderManager()
    store = DummyTenantStore((db_tenant,))
    redis = EventRedis([{"type": "message", "data": b"7"}])
    app = SimpleNamespace(
        state=SimpleNamespace(
            config_holder=holder,
            provider_manager=manager,
            tenant_store=store,
            task_registry=SimpleNamespace(create=lambda *args, **kwargs: None),
        )
    )

    await _watch_config_events_once(app, redis)  # type: ignore[arg-type]

    assert holder.version == 7
    assert holder.current.tenants == (db_tenant,)
    assert redis.pubsub_client.subscribed == [CFG_EVENTS_CHANNEL]
    assert redis.pubsub_client.unsubscribed == [CFG_EVENTS_CHANNEL]
    assert redis.pubsub_client.closed


@pytest.mark.asyncio
async def test_poll_config_version_recovers_after_redis_error(poll_config_path: Path) -> None:
    holder = ConfigHolder(load_config(poll_config_path), poll_config_path)
    redis = FlakyRedis()
    redis.values[CFG_VERSION_KEY] = b"1"
    app = SimpleNamespace(
        state=SimpleNamespace(
            config_holder=holder,
            provider_manager=DummyProviderManager(),
            task_registry=SimpleNamespace(create=lambda *args, **kwargs: None),
        )
    )

    task = asyncio.create_task(poll_config_version(app, redis))  # type: ignore[arg-type]
    try:
        for _ in range(100):
            if holder.version == 1:
                break
            await asyncio.sleep(0.01)
        assert holder.version == 1
        assert redis.get_calls >= 2
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.fixture
def poll_config_path(tmp_path: Path) -> Path:
    source = Path("tests/fixtures/gateway-test.yaml")
    config_path = tmp_path / "gateway.yaml"
    raw = (
        source.read_text(encoding="utf-8")
        .replace("config_poll_interval_s: 30.0", "config_poll_interval_s: 0.01")
        .replace("config_reload_backoff_initial_s: 0.1", "config_reload_backoff_initial_s: 0.01")
        .replace("config_reload_backoff_max_s: 1.0", "config_reload_backoff_max_s: 0.02")
    )
    config_path.write_text(raw, encoding="utf-8")
    return config_path
