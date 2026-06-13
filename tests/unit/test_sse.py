from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from tidegate.api.sse import stream_chunk_payload, with_heartbeats
from tidegate.core.models import UnifiedDelta, Usage


async def _slow_delta() -> AsyncIterator[UnifiedDelta]:
    await asyncio.sleep(0.05)
    yield UnifiedDelta(content="tok0 ")


@pytest.mark.asyncio
async def test_with_heartbeats_emits_idle_marker() -> None:
    events = []
    async for event in with_heartbeats(_slow_delta(), 0.01):
        events.append(event)
    assert events[0] is None
    assert events[-1] == UnifiedDelta(content="tok0 ")


def test_stream_chunk_payload_usage_gate() -> None:
    delta = UnifiedDelta(usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3))
    assert stream_chunk_payload("req", "chat-large", delta, include_usage=False) is None
    payload = stream_chunk_payload("req", "chat-large", delta, include_usage=True)
    assert payload is not None
    assert payload["id"] == "req"
    assert payload["model"] == "chat-large"
    assert payload["choices"] == []
    assert payload["usage"] == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
