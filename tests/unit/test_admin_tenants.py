from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
from starlette.requests import Request
from starlette.responses import Response

from tidegate.api.admin import get_tenant, upsert_tenant
from tidegate.config.holder import ConfigHolder
from tidegate.config.loader import load_config
from tidegate.config.models import CacheToggleConfig, TenantConfig
from tidegate.config.tenant_store import TenantStore


@pytest.mark.asyncio
async def test_get_tenant_requires_admin_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIDEGATE_ADMIN_TOKEN", "dev-admin")
    request = _request(headers={"x-admin-token": "wrong"})

    response = await get_tenant(request, "demo")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_put_tenant_returns_503_without_postgres(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIDEGATE_ADMIN_TOKEN", "dev-admin")
    settings = load_config("tests/fixtures/gateway-test.yaml")
    request = _request(
        headers={"x-admin-token": "dev-admin"},
        state=SimpleNamespace(
            tenant_store=TenantStore(None),
            config_holder=ConfigHolder(settings, settings_path()),
        ),
        body=_tenant_payload(),
    )

    response = await upsert_tenant(request, "demo")

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_put_tenant_validates_references(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIDEGATE_ADMIN_TOKEN", "dev-admin")
    settings = load_config("tests/fixtures/gateway-test.yaml")
    request = _request(
        headers={"x-admin-token": "dev-admin"},
        state=SimpleNamespace(
            tenant_store=_MemoryTenantStore(()),
            config_holder=ConfigHolder(settings, settings_path()),
        ),
        body=_tenant_payload(plan="missing"),
    )

    response = await upsert_tenant(request, "demo")

    assert response.status_code == 422
    assert b"unknown quota plan" in response.body


@pytest.mark.asyncio
async def test_put_tenant_rejects_l2_when_runtime_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TIDEGATE_ADMIN_TOKEN", "dev-admin")
    settings = load_config("tests/fixtures/gateway-test.yaml")
    request = _request(
        headers={"x-admin-token": "dev-admin"},
        state=SimpleNamespace(
            tenant_store=_MemoryTenantStore(()),
            config_holder=ConfigHolder(settings, settings_path()),
            embedding_service=None,
        ),
        body=_tenant_payload(
            cache=CacheToggleConfig(l1=True, l2=True, l2_operating_point="conservative"),
        ),
    )

    response = await upsert_tenant(request, "demo")

    assert response.status_code == 422
    assert b"l2 cache runtime unavailable" in response.body


@pytest.mark.asyncio
async def test_put_tenant_upserts_replaces_snapshot_and_publishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TIDEGATE_ADMIN_TOKEN", "dev-admin")
    settings = load_config("tests/fixtures/gateway-test.yaml")
    holder = ConfigHolder(settings, settings_path())
    tenant = TenantConfig(
        id="demo",
        api_key_sha256="new-key",
        plan="free",
        policy="default",
        cache=CacheToggleConfig(l1=False, l2=False),
    )
    store = _MemoryTenantStore((settings.tenants[0],))
    redis = _FakeRedis()
    request = _request(
        headers={"x-admin-token": "dev-admin"},
        state=SimpleNamespace(
            tenant_store=store,
            config_holder=holder,
            redis=redis,
        ),
        body=_tenant_payload(api_key_sha256=tenant.api_key_sha256, cache=tenant.cache),
    )

    response = await upsert_tenant(request, "demo")

    assert response.status_code == 200
    assert json.loads(_response_body(response))["version"] == 1
    assert holder.version == 1
    assert holder.current.tenants == (tenant,)
    assert store.upserts == [tenant]
    assert redis.published == [("cfg:events", "1")]


@pytest.mark.asyncio
async def test_get_tenant_reads_from_current_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIDEGATE_ADMIN_TOKEN", "dev-admin")
    settings = load_config("tests/fixtures/gateway-test.yaml")
    request = _request(
        headers={"x-admin-token": "dev-admin"},
        state=SimpleNamespace(config_holder=ConfigHolder(settings, settings_path())),
    )

    response = await get_tenant(request, "demo")

    assert response.status_code == 200
    assert json.loads(_response_body(response))["tenant"]["id"] == "demo"


def settings_path() -> Any:
    return "tests/fixtures/gateway-test.yaml"


def _response_body(response: Response) -> bytes:
    body = response.body
    if isinstance(body, memoryview):
        return body.tobytes()
    return bytes(body)


def _tenant_payload(
    *,
    api_key_sha256: str = "new-key",
    plan: str = "free",
    policy: str = "default",
    cache: CacheToggleConfig | None = None,
) -> bytes:
    return json.dumps(
        {
            "api_key_sha256": api_key_sha256,
            "plan": plan,
            "policy": policy,
            "cache": (cache or CacheToggleConfig()).model_dump(),
        }
    ).encode()


def _request(
    *,
    headers: dict[str, str] | None = None,
    state: object | None = None,
    body: bytes = b"",
) -> Request:
    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": body, "more_body": False}

    app = SimpleNamespace(state=state or SimpleNamespace())
    scope: dict[str, object] = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [
            (key.lower().encode(), value.encode()) for key, value in (headers or {}).items()
        ],
        "app": app,
        "state": {},
    }
    return Request(cast(Any, scope), receive=receive)


class _FakeRedis:
    def __init__(self) -> None:
        self.value = 0
        self.published: list[tuple[str, str]] = []

    async def incr(self, key: str) -> int:
        assert key == "cfg:version"
        self.value += 1
        return self.value

    async def publish(self, channel: str, value: str) -> None:
        self.published.append((channel, value))


class _MemoryTenantStore:
    enabled = True

    def __init__(self, tenants: tuple[TenantConfig, ...]) -> None:
        self.tenants = tenants
        self.upserts: list[TenantConfig] = []

    async def upsert(self, tenant: TenantConfig) -> None:
        self.upserts.append(tenant)
        self.tenants = (tenant,)

    async def load_all(self) -> tuple[TenantConfig, ...]:
        return self.tenants
