from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import structlog
from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from tidegate.core.models import UnifiedDelta, Usage


@dataclass
class StreamAccounting:
    content_chars: int = 0
    delta_count: int = 0
    ttft_ms: float | None = None
    usage: Usage | None = None
    started_at: float = 0.0


class DisconnectAwareStreamingResponse(StreamingResponse):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await super().__call__(scope, receive, send)
            return

        # Starlette's ASGI 2.4 path no longer listens for disconnects by default.
        stream_task = asyncio.create_task(self.stream_response(send))
        disconnect_task = asyncio.create_task(self.listen_for_disconnect(receive))
        done, pending = await asyncio.wait(
            {stream_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        if disconnect_task in done and not stream_task.done():
            stream_task.cancel()
            await asyncio.gather(stream_task, return_exceptions=True)
            await _close_iterator(self.body_iterator)
        elif stream_task.done():
            await stream_task

        if self.background is not None:
            await self.background()


async def _close_iterator(iterator: Any) -> None:
    aclose = getattr(iterator, "aclose", None)
    if callable(aclose):
        await aclose()


def sse_event(payload: dict[str, object] | str) -> bytes:
    if isinstance(payload, str):
        return f"data: {payload}\n\n".encode()
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def heartbeat_event() -> bytes:
    return b": ping\n\n"


def stream_chunk_payload(
    request_id: str,
    model: str,
    delta: UnifiedDelta,
    *,
    include_usage: bool,
) -> dict[str, object] | None:
    base: dict[str, object] = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
    }
    if delta.usage is not None:
        if not include_usage:
            return None
        return base | {"choices": [], "usage": delta.usage.model_dump()}
    if delta.finish_reason is not None:
        return base | {
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": delta.finish_reason},
            ]
        }
    return base | {
        "choices": [
            {
                "index": 0,
                "delta": {"content": delta.content or ""},
                "finish_reason": None,
            },
        ]
    }


async def with_heartbeats(
    deltas: AsyncIterator[UnifiedDelta],
    heartbeat_interval_s: float,
) -> AsyncIterator[UnifiedDelta | None]:
    # Yield None as an SSE heartbeat slot while upstream is idle.
    pending: asyncio.Task[UnifiedDelta] = asyncio.create_task(_next_delta(deltas))
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=heartbeat_interval_s)
            if pending in done:
                try:
                    yield pending.result()
                except StopAsyncIteration:
                    return
                pending = asyncio.create_task(_next_delta(deltas))
            else:
                yield None
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        aclose = getattr(deltas, "aclose", None)
        if callable(aclose):
            await aclose()


async def _next_delta(deltas: AsyncIterator[UnifiedDelta]) -> UnifiedDelta:
    return await anext(deltas)


async def error_chunk(request_id: str, model: str) -> AsyncIterator[bytes]:
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
    }
    yield sse_event(payload)
    yield sse_event("[DONE]")


def log_stream_access(
    *,
    tenant: str,
    model: str,
    outcome: str,
    route: str,
    accounting: StreamAccounting,
    duration_ms: float,
    trace_id: str | None = None,
) -> None:
    structlog.get_logger().info(
        "access",
        tenant=tenant,
        model=model,
        outcome=outcome,
        route=route,
        ttft_ms=accounting.ttft_ms,
        duration_ms=duration_ms,
        forwarded_chars=accounting.content_chars,
        forwarded_deltas=accounting.delta_count,
        usage=None if accounting.usage is None else accounting.usage.model_dump(),
        trace_id=trace_id,
    )
