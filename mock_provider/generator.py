from __future__ import annotations

import asyncio
import json
import math
import random
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, cast

from fastapi import HTTPException, Request

MockFail = Literal["none", "error_500", "error_429", "timeout", "stall_after_n", "drop_after_n"]


@dataclass(frozen=True)
class MockDirective:
    ttft_ms: int
    tpot_ms: int
    output_tokens: int
    fail: MockFail
    fail_n: int
    retry_after_s: int


@dataclass(frozen=True)
class MockDefaults:
    ttft_ms: int
    tpot_ms: int
    output_tokens: int
    ttft_lognorm_mu: float | None = None
    ttft_lognorm_sigma: float | None = None


def directive_from_header(request: Request, defaults: MockDefaults) -> MockDirective:
    return directive_from_mapping(request.headers.get("x-mock-directive"), defaults)


def directive_from_mapping(
    raw: str | dict[str, object] | None, defaults: MockDefaults
) -> MockDirective:
    base_ttft = defaults.ttft_ms
    if defaults.ttft_lognorm_mu is not None and defaults.ttft_lognorm_sigma is not None:
        base_ttft = max(
            0,
            math.floor(
                random.lognormvariate(defaults.ttft_lognorm_mu, defaults.ttft_lognorm_sigma)
            ),
        )

    directive = MockDirective(
        ttft_ms=base_ttft,
        tpot_ms=defaults.tpot_ms,
        output_tokens=defaults.output_tokens,
        fail="none",
        fail_n=10,
        retry_after_s=5,
    )
    if raw is None:
        return directive

    parsed = json.loads(raw) if isinstance(raw, str) else raw
    fail = str(parsed.get("fail", directive.fail))
    if fail not in {
        "none",
        "error_500",
        "error_429",
        "timeout",
        "stall_after_n",
        "drop_after_n",
    }:
        raise HTTPException(status_code=422, detail="invalid mock fail directive")
    return MockDirective(
        ttft_ms=_int_field(parsed, "ttft_ms", directive.ttft_ms),
        tpot_ms=_int_field(parsed, "tpot_ms", directive.tpot_ms),
        output_tokens=_int_field(parsed, "output_tokens", directive.output_tokens),
        fail=cast(MockFail, fail),
        fail_n=_int_field(parsed, "fail_n", directive.fail_n),
        retry_after_s=_int_field(parsed, "retry_after_s", directive.retry_after_s),
    )


def _int_field(parsed: dict[str, object], key: str, default: int) -> int:
    value = parsed.get(key, default)
    if isinstance(value, bool):
        raise HTTPException(status_code=422, detail=f"invalid mock integer field: {key}")
    if isinstance(value, int | float | str):
        return int(value)
    raise HTTPException(status_code=422, detail=f"invalid mock integer field: {key}")


def prompt_tokens(body: dict[str, object]) -> int:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return 0
    total_chars = 0
    for message in messages:
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                total_chars += len(content)
    return max(1, math.ceil(total_chars / 4))


def completion_text(output_tokens: int) -> str:
    return "".join(f"tok{i} " for i in range(output_tokens))


def chat_completion_response(
    body: dict[str, object], directive: MockDirective
) -> dict[str, object]:
    content = completion_text(directive.output_tokens)
    prompt = prompt_tokens(body)
    return {
        "id": f"chatcmpl-mock-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": str(body.get("model", "mock-model")),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": directive.output_tokens,
            "total_tokens": prompt + directive.output_tokens,
        },
    }


def sse_payload(payload: dict[str, object] | str) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


async def stream_chunks(
    body: dict[str, object],
    directive: MockDirective,
    request: Request,
) -> AsyncIterator[str]:
    await asyncio.sleep(directive.ttft_ms / 1000)
    if directive.fail == "timeout":
        await asyncio.Event().wait()

    model = str(body.get("model", "mock-model"))
    prompt = prompt_tokens(body)
    base = {
        "id": f"chatcmpl-mock-{int(time.time() * 1000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
    }
    for index in range(directive.output_tokens):
        if directive.fail == "stall_after_n" and index >= directive.fail_n:
            await asyncio.Event().wait()
        if directive.fail == "drop_after_n" and index >= directive.fail_n:
            raise RuntimeError("mock provider dropped stream")
        if await request.is_disconnected():
            raise asyncio.CancelledError
        payload = base | {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": f"tok{index} "},
                    "finish_reason": None,
                }
            ]
        }
        yield sse_payload(payload)
        await asyncio.sleep(directive.tpot_ms / 1000)

    yield sse_payload(base | {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    yield sse_payload(
        base
        | {
            "choices": [],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": directive.output_tokens,
                "total_tokens": prompt + directive.output_tokens,
            },
        }
    )
    yield sse_payload("[DONE]")
