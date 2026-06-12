from __future__ import annotations


def exact_key(tenant_id: str, digest: str) -> str:
    return f"cache:exact:{tenant_id}:{digest}"


def semcache_key(entry_id: str) -> str:
    return f"semcache:{entry_id}"
