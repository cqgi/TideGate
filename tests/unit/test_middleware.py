from __future__ import annotations

import hashlib

from tidegate.api.middleware import _AuthCache
from tidegate.config.models import TenantConfig


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
