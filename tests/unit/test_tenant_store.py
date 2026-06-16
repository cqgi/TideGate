from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from tidegate.config.models import CacheToggleConfig, TenantConfig
from tidegate.config.tenant_store import TenantStore


@pytest.mark.asyncio
async def test_tenant_store_noops_without_postgres() -> None:
    store = TenantStore(None)
    tenant = _tenant("demo")

    await store.ensure_schema()
    await store.seed_from_yaml((tenant,))

    assert not store.enabled
    assert await store.load_all() is None
    assert await store.max_updated_at() is None
    with pytest.raises(RuntimeError, match="tenant config database unavailable"):
        await store.upsert(tenant)


@pytest.mark.asyncio
async def test_seed_from_yaml_inserts_missing_tenants_only() -> None:
    pool = _FakePool()
    store = TenantStore(cast(Any, pool))
    db_tenant = _tenant(
        "demo",
        api_key_sha256="db-key",
        plan="paid",
        cache=CacheToggleConfig(l1=False, l2=True, l2_operating_point="balanced"),
    )
    yaml_tenant = _tenant(
        "demo",
        api_key_sha256="yaml-key",
        plan="free",
        cache=CacheToggleConfig(l1=True, l2=False),
    )
    other_tenant = _tenant("other", api_key_sha256="other-key")

    await store.ensure_schema()
    await store.upsert(db_tenant)
    await store.seed_from_yaml((yaml_tenant, other_tenant))

    tenants = await store.load_all()
    assert tenants == (
        db_tenant,
        other_tenant,
    )
    assert pool.schema_created


@pytest.mark.asyncio
async def test_upsert_updates_existing_tenant_and_max_updated_at() -> None:
    pool = _FakePool()
    store = TenantStore(cast(Any, pool))
    initial = _tenant("demo", api_key_sha256="old-key")
    updated = _tenant(
        "demo",
        api_key_sha256="new-key",
        policy="premium-fast",
        cache=CacheToggleConfig(l1=True, l2=True, l2_operating_point="conservative"),
    )

    await store.upsert(initial)
    first_updated_at = await store.max_updated_at()
    await store.upsert(updated)

    assert await store.load_all() == (updated,)
    assert first_updated_at is not None
    assert await store.max_updated_at() == first_updated_at + timedelta(seconds=1)


def _tenant(
    tenant_id: str,
    *,
    api_key_sha256: str | None = None,
    plan: str = "free",
    policy: str = "default",
    cache: CacheToggleConfig | None = None,
) -> TenantConfig:
    return TenantConfig(
        id=tenant_id,
        api_key_sha256=api_key_sha256 or f"{tenant_id}-key",
        plan=plan,
        policy=policy,
        cache=cache or CacheToggleConfig(),
    )


class _FakePool:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, object]] = {}
        self.schema_created = False
        self._clock = datetime(2026, 1, 1, tzinfo=UTC)

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self)

    def next_updated_at(self) -> datetime:
        value = self._clock
        self._clock += timedelta(seconds=1)
        return value


class _FakeAcquire:
    def __init__(self, pool: _FakePool) -> None:
        self._pool = pool

    async def __aenter__(self) -> _FakeConnection:
        return _FakeConnection(self._pool)

    async def __aexit__(self, *exc_info: object) -> None:
        return None


class _FakeConnection:
    def __init__(self, pool: _FakePool) -> None:
        self._pool = pool

    async def execute(self, sql: str, *args: object) -> str:
        if not args:
            self._pool.schema_created = True
            return "CREATE TABLE"
        self._upsert(args)
        return "INSERT 0 1"

    async def executemany(self, sql: str, rows: Iterable[Sequence[object]]) -> None:
        del sql
        for row in rows:
            tenant_id = str(row[0])
            if tenant_id not in self._pool.rows:
                self._insert(row)

    async def fetch(self, sql: str) -> list[dict[str, object]]:
        del sql
        return [self._pool.rows[tenant_id] for tenant_id in sorted(self._pool.rows)]

    async def fetchval(self, sql: str) -> object | None:
        del sql
        if not self._pool.rows:
            return None
        return max(cast(datetime, row["updated_at"]) for row in self._pool.rows.values())

    def _insert(self, row: Sequence[object]) -> None:
        self._pool.rows[str(row[0])] = {
            "tenant_id": row[0],
            "api_key_sha256": row[1],
            "plan": row[2],
            "policy": row[3],
            "cache": row[4],
            "updated_at": self._pool.next_updated_at(),
        }

    def _upsert(self, row: Sequence[object]) -> None:
        self._insert(row)
