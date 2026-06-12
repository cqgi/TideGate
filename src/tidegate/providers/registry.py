from __future__ import annotations

import os
from collections.abc import Callable

from tidegate.config.models import GatewayConfig, ProviderConfig
from tidegate.providers.base import Provider

ProviderFactory = Callable[[str, ProviderConfig], Provider]

_REGISTRY: dict[str, ProviderFactory] = {}


def register_provider(provider_type: str) -> Callable[[ProviderFactory], ProviderFactory]:
    def decorator(factory: ProviderFactory) -> ProviderFactory:
        _REGISTRY[provider_type] = factory
        return factory

    return decorator


def build_providers(config: GatewayConfig) -> dict[str, Provider]:
    from tidegate.providers import openai_compat  # noqa: F401

    providers: dict[str, Provider] = {}
    for name, provider_config in config.providers.items():
        factory = _REGISTRY[provider_config.type]
        # SPEC-M1-3: each provider instance owns its httpx connection pool.
        providers[name] = factory(name, provider_config)
    return providers


def api_key_from_env(env_name: str) -> str:
    return os.getenv(env_name, "")
