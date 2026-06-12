from __future__ import annotations

import asyncio
from collections.abc import Mapping

from tidegate.config.models import GatewayConfig
from tidegate.providers.base import Provider
from tidegate.providers.registry import build_provider, build_providers


class ProviderManager:
    def __init__(self, settings: GatewayConfig) -> None:
        self._providers = build_providers(settings)

    @property
    def providers(self) -> Mapping[str, Provider]:
        return self._providers

    async def close(self) -> None:
        await asyncio.gather(
            *(provider.aclose() for provider in self._providers.values()),
            return_exceptions=True,
        )

    def rebuild_if_needed(self, previous: GatewayConfig, current: GatewayConfig) -> list[Provider]:
        if previous.providers == current.providers:
            return []
        next_providers: dict[str, Provider] = {}
        old_to_close: list[Provider] = []
        for name, provider_config in current.providers.items():
            if previous.providers.get(name) == provider_config and name in self._providers:
                next_providers[name] = self._providers[name]
            else:
                # SPEC-M1-4: only changed/new provider instances get rebuilt.
                next_providers[name] = build_provider(name, provider_config)
        for name, provider in self._providers.items():
            if current.providers.get(name) != previous.providers.get(name):
                old_to_close.append(provider)
        self._providers = next_providers
        return old_to_close


async def close_later(providers: list[Provider], delay_s: float) -> None:
    if delay_s > 0:
        await asyncio.sleep(delay_s)
    await asyncio.gather(*(provider.aclose() for provider in providers), return_exceptions=True)
