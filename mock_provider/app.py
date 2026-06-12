from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response

from mock_provider.generator import (
    MockDefaults,
    MockDirective,
    chat_completion_response,
    directive_from_header,
    directive_from_mapping,
    stream_chunks,
)


@dataclass
class ProviderStats:
    started: int = 0
    completed: int = 0
    aborted: int = 0
    active: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "started": self.started,
            "completed": self.completed,
            "aborted": self.aborted,
            "active": self.active,
        }

    def reset(self) -> None:
        self.started = 0
        self.completed = 0
        self.aborted = 0
        self.active = 0


def create_app(defaults: MockDefaults) -> FastAPI:
    app = FastAPI(title="TideGate Mock Provider")
    stats = ProviderStats()
    behavior: dict[str, object] | None = None

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request) -> Response:
        nonlocal behavior
        body = await request.json()
        directive = (
            directive_from_header(request, defaults)
            if request.headers.get("x-mock-directive") is not None
            else directive_from_mapping(behavior, defaults)
        )
        if directive.fail == "error_500":
            raise HTTPException(status_code=500, detail="mock upstream error")
        if directive.fail == "error_429":
            raise HTTPException(
                status_code=429,
                detail="mock upstream rate limited",
                headers={"Retry-After": str(directive.retry_after_s)},
            )

        # SPEC-M0-1: expose deterministic OpenAI-compatible generation plus abort stats.
        stats.started += 1
        if body.get("stream") is True:
            return StreamingResponse(
                _tracked_stream(body, directive, request, stats),
                media_type="text/event-stream",
            )

        stats.active += 1
        try:
            if directive.fail == "timeout":
                await asyncio.Event().wait()
            await asyncio.sleep(directive.ttft_ms / 1000)
            response = chat_completion_response(body, directive)
            stats.completed += 1
            return JSONResponse(response)
        finally:
            stats.active -= 1

    @app.get("/__stats")
    async def get_provider_stats() -> dict[str, int]:
        return stats.snapshot()

    @app.post("/__reset")
    async def reset_provider_stats() -> dict[str, int]:
        nonlocal behavior
        stats.reset()
        behavior = None
        return stats.snapshot()

    @app.post("/__behavior")
    async def set_behavior(request: Request) -> dict[str, object]:
        nonlocal behavior
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="behavior must be an object")
        directive = directive_from_mapping(payload, defaults)
        behavior = {
            "ttft_ms": directive.ttft_ms,
            "tpot_ms": directive.tpot_ms,
            "output_tokens": directive.output_tokens,
            "fail": directive.fail,
            "fail_n": directive.fail_n,
            "retry_after_s": directive.retry_after_s,
            "logprob_mean": directive.logprob_mean,
        }
        return {"ok": True, "behavior": behavior}

    return app


async def _tracked_stream(
    body: dict[str, object],
    directive: MockDirective,
    request: Request,
    stats: ProviderStats,
) -> AsyncIterator[str]:
    stats.active += 1
    completed = False
    aborted = False
    try:
        # SPEC-M0-1: incomplete streams count as aborted for disconnect verification.
        async for chunk in stream_chunks(body, directive, request):
            yield chunk
        completed = True
        stats.completed += 1
    except asyncio.CancelledError:
        aborted = True
        raise
    finally:
        if aborted or not completed:
            stats.aborted += 1
        stats.active -= 1
