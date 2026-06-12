from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass

from tidegate.config.models import GatewayConfig
from tidegate.providers.base import Provider
from tidegate.providers.registry import build_providers


@dataclass(frozen=True)
class ProviderSnapshot:
    providers: Mapping[str, Provider]


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
        # DECISION: M1 rebuilds the full provider map on provider config changes; simple and safe.
        old = list(self._providers.values())
        self._providers = build_providers(current)
        return old


async def close_later(providers: list[Provider], delay_s: float) -> None:
    if delay_s > 0:
        await asyncio.sleep(delay_s)
    await asyncio.gather(*(provider.aclose() for provider in providers), return_exceptions=True)
