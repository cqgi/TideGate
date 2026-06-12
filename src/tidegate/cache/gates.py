from __future__ import annotations

from dataclasses import dataclass

from tidegate.config.models import GatewayConfig, TenantConfig
from tidegate.core.models import UnifiedRequest, UnifiedResponse


@dataclass(frozen=True)
class CacheReadDecision:
    l1: bool
    l2: bool
    bypass_reason: str | None = None


def read_decision(
    req: UnifiedRequest,
    tenant: TenantConfig,
    settings: GatewayConfig,
) -> CacheReadDecision:
    if _is_volatile(req, settings):
        return CacheReadDecision(False, False, "volatile")
    if req.has_tools:
        return CacheReadDecision(False, False, "tools")
    l1 = tenant.cache.l1
    l2 = tenant.cache.l2 and _temperature_ok(req, settings) and l2_context_allowed(req)
    return CacheReadDecision(l1, l2)


def can_store(
    req: UnifiedRequest,
    tenant: TenantConfig,
    response: UnifiedResponse,
    settings: GatewayConfig,
    *,
    degraded: bool,
) -> tuple[bool, bool]:
    if degraded or req.has_tools or _is_volatile(req, settings):
        return False, False
    if response.finish_reason != "stop":
        return False, False
    content = response.content.strip()
    if not content:
        return False, False
    if len(content.encode("utf-8")) > settings.cache.l1.max_value_bytes:
        return False, False
    if any(pattern in content for pattern in settings.cache.reject_patterns):
        return False, False
    l1 = tenant.cache.l1
    l2 = tenant.cache.l2 and _temperature_ok(req, settings) and l2_context_allowed(req)
    return l1, l2


def l2_context_allowed(req: UnifiedRequest) -> bool:
    return sum(1 for message in req.messages if message.role == "user") == 1


def _temperature_ok(req: UnifiedRequest, settings: GatewayConfig) -> bool:
    temperature = 0.0 if req.temperature is None else req.temperature
    return temperature <= settings.cache.l2.max_temperature


def _is_volatile(req: UnifiedRequest, settings: GatewayConfig) -> bool:
    text = "\n".join(message.content or "" for message in req.messages)
    return any(pattern in text for pattern in settings.cache.volatile_intent_patterns)
