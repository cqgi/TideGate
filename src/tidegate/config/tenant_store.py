from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any

import asyncpg

from tidegate.config.models import CacheToggleConfig, TenantConfig

TENANT_CONFIG_DDL = """
CREATE TABLE IF NOT EXISTS tenant_configs (
    tenant_id       TEXT PRIMARY KEY,
    api_key_sha256  TEXT UNIQUE NOT NULL,
    plan            TEXT NOT NULL,
    policy          TEXT NOT NULL,
    cache           JSONB NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


class TenantStore:
    def __init__(self, pg_pool: asyncpg.Pool | None) -> None:
        self._pg_pool = pg_pool

    @property
    def enabled(self) -> bool:
        return self._pg_pool is not None

    async def ensure_schema(self) -> None:
        if self._pg_pool is None:
            return
        async with self._pg_pool.acquire() as conn:
            await conn.execute(TENANT_CONFIG_DDL)

    async def seed_from_yaml(self, yaml_tenants: tuple[TenantConfig, ...]) -> None:
        if self._pg_pool is None or not yaml_tenants:
            return
        rows = [_tenant_row(tenant) for tenant in yaml_tenants]
        async with self._pg_pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO tenant_configs (
                  tenant_id, api_key_sha256, plan, policy, cache
                )
                VALUES ($1, $2, $3, $4, $5::jsonb)
                ON CONFLICT (tenant_id) DO NOTHING
                """,
                rows,
            )

    async def load_all(self) -> tuple[TenantConfig, ...] | None:
        if self._pg_pool is None:
            return None
        async with self._pg_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT tenant_id, api_key_sha256, plan, policy, cache
                FROM tenant_configs
                ORDER BY tenant_id
                """
            )
        return tuple(_tenant_from_row(row) for row in rows)

    async def upsert(self, tenant: TenantConfig) -> None:
        if self._pg_pool is None:
            raise RuntimeError("tenant config database unavailable")
        async with self._pg_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO tenant_configs (
                  tenant_id, api_key_sha256, plan, policy, cache
                )
                VALUES ($1, $2, $3, $4, $5::jsonb)
                ON CONFLICT (tenant_id) DO UPDATE SET
                  api_key_sha256 = EXCLUDED.api_key_sha256,
                  plan = EXCLUDED.plan,
                  policy = EXCLUDED.policy,
                  cache = EXCLUDED.cache,
                  updated_at = now()
                """,
                *_tenant_row(tenant),
            )

    async def max_updated_at(self) -> datetime | None:
        if self._pg_pool is None:
            return None
        async with self._pg_pool.acquire() as conn:
            value = await conn.fetchval("SELECT max(updated_at) FROM tenant_configs")
        return value if isinstance(value, datetime) else None


def _tenant_row(tenant: TenantConfig) -> tuple[str, str, str, str, str]:
    return (
        tenant.id,
        tenant.api_key_sha256,
        tenant.plan,
        tenant.policy,
        json.dumps(tenant.cache.model_dump(), sort_keys=True),
    )


def _tenant_from_row(row: Mapping[str, Any]) -> TenantConfig:
    return TenantConfig(
        id=row["tenant_id"],
        api_key_sha256=row["api_key_sha256"],
        plan=row["plan"],
        policy=row["policy"],
        cache=_cache_from_db(row["cache"]),
    )


def _cache_from_db(value: object) -> CacheToggleConfig:
    if isinstance(value, str):
        raw = json.loads(value)
    elif isinstance(value, bytes | bytearray):
        raw = json.loads(value.decode())
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise ValueError("tenant cache config must be a JSON object")
    return CacheToggleConfig.model_validate(raw)
