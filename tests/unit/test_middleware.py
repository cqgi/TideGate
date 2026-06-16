from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from tidegate.api.middleware import AuthMiddleware, _AuthCache
from tidegate.config.holder import ConfigHolder
from tidegate.config.loader import load_config
from tidegate.config.models import CacheToggleConfig, TenantConfig


def test_auth_cache_ttl_expiry() -> None:
    tenant = TenantConfig(id="demo", api_key_sha256=hashlib.sha256(b"k").hexdigest())
    cache = _AuthCache(capacity=2, ttl_s=-1.0)
    cache.put("k", tenant)
    assert cache.get("k") is None


def test_auth_cache_capacity_evicts_lru() -> None:
    first = TenantConfig(id="first", api_key_sha256=hashlib.sha256(b"1").hexdigest())
    second = TenantConfig(id="second", api_key_sha256=hashlib.sha256(b"2").hexdigest())
    cache = _AuthCache(capacity=1, ttl_s=60.0)
    cache.put("1", first)
    cache.put("2", second)
    assert cache.get("1") is None
    assert cache.get("2") == second


def test_auth_cache_invalidate_tenant_keeps_other_tenants() -> None:
    first = TenantConfig(id="first", api_key_sha256=hashlib.sha256(b"1").hexdigest())
    second = TenantConfig(id="second", api_key_sha256=hashlib.sha256(b"2").hexdigest())
    cache = _AuthCache(capacity=4, ttl_s=60.0)

    cache.put("1", first)
    cache.put("2", second)
    cache.invalidate_tenant("first")

    assert cache.get("1") is None
    assert cache.get("2") == second


def test_auth_middleware_only_invalidates_changed_tenants_on_version_change() -> None:
    base = load_config("tests/fixtures/gateway-test.yaml")
    first = TenantConfig(id="first", api_key_sha256=hashlib.sha256(b"1").hexdigest())
    second = TenantConfig(id="second", api_key_sha256=hashlib.sha256(b"2").hexdigest())
    updated_first = first.model_copy(update={"cache": CacheToggleConfig(l1=False, l2=False)})
    holder = ConfigHolder(
        base.model_copy(update={"tenants": (first, second)}),
        Path("tests/fixtures/gateway-test.yaml"),
    )
    middleware = AuthMiddleware(SimpleNamespace(), holder)  # type: ignore[arg-type]
    middleware._cache.put("1", first)
    middleware._cache.put("2", second)

    holder.replace(
        base.model_copy(update={"tenants": (updated_first, second)}),
        version=holder.version + 1,
    )
    middleware._reset_cache_if_config_changed()

    assert middleware._cache.get("1") is None
    assert middleware._cache.get("2") == second
