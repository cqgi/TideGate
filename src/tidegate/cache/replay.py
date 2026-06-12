from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from tidegate.config.models import GatewayConfig
from tidegate.core.models import UnifiedDelta, UnifiedResponse


async def replay_as_stream(
    response: UnifiedResponse,
    settings: GatewayConfig,
) -> AsyncIterator[UnifiedDelta]:
    chunk_chars = settings.cache.replay_chunk_chars
    interval_s = settings.cache.replay_interval_ms / 1000
    for index in range(0, len(response.content), chunk_chars):
        yield UnifiedDelta(content=response.content[index : index + chunk_chars])
        if interval_s > 0:
            await asyncio.sleep(interval_s)
    yield UnifiedDelta(finish_reason=response.finish_reason)
    yield UnifiedDelta(usage=response.usage)
