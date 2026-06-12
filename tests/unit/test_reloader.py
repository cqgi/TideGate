from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tidegate.config.holder import ConfigHolder
from tidegate.config.loader import load_config
from tidegate.config.reloader import (
    CFG_EVENTS_CHANNEL,
    CFG_VERSION_KEY,
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


class DummyProviderManager:
    def rebuild_if_needed(self, previous: object, current: object) -> list[Any]:
        del previous, current
        return []


@pytest.mark.asyncio
async def test_publish_reload_increments_and_publishes() -> None:
    """SPEC-M1-4."""
    redis = FakeRedis()
    version = await publish_reload(redis)  # type: ignore[arg-type]
    assert version == 1
    assert redis.values[CFG_VERSION_KEY] == b"1"
    assert redis.published == [(CFG_EVENTS_CHANNEL, "1")]


@pytest.mark.asyncio
async def test_poll_config_version_applies_new_version(poll_config_path: Path) -> None:
    """SPEC-M1-4."""
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


@pytest.fixture
def poll_config_path(tmp_path: Path) -> Path:
    source = Path("tests/fixtures/gateway-test.yaml")
    config_path = tmp_path / "gateway.yaml"
    raw = source.read_text(encoding="utf-8").replace(
        "config_poll_interval_s: 30.0", "config_poll_interval_s: 0.01"
    )
    config_path.write_text(raw, encoding="utf-8")
    return config_path
