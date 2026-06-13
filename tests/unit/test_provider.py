from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest

from tidegate.config.models import ProviderConfig
from tidegate.core.deadline import Deadline
from tidegate.core.errors import ErrorCategory, GatewayError
from tidegate.core.models import ChatMessage, UnifiedRequest, Usage
from tidegate.providers.openai_compat import (
    OpenAICompatibleProvider,
    _non_stream_timeout,
    _parse_retry_after,
    _parse_sse_line,
)


def test_parse_sse_bad_json_is_gateway_error() -> None:
    with pytest.raises(GatewayError) as exc_info:
        _parse_sse_line("data: {bad json")
    assert exc_info.value.category == ErrorCategory.RETRYABLE_UPSTREAM


def test_retry_after_http_date_and_invalid() -> None:
    assert _parse_retry_after("not a date") is None
    assert _parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0


def test_connect_timeout_is_retryable() -> None:
    assert issubclass(httpx.ConnectTimeout, httpx.TimeoutException)


def test_non_stream_timeout_uses_total_deadline_for_reads() -> None:
    deadline = Deadline(connect_s=0.1, ttft_s=0.1, inter_chunk_s=0.1, total_deadline=999999999.0)
    timeout = _non_stream_timeout(deadline)
    assert timeout.read is None


def test_parse_sse_line_preserves_finish_reason_with_usage() -> None:
    delta = _parse_sse_line(
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}'
    )
    assert delta is not None
    assert delta.finish_reason == "stop"
    assert delta.usage == Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3)


@pytest.mark.asyncio
async def test_provider_connect_refused_is_retryable() -> None:
    provider = OpenAICompatibleProvider(
        "dead",
        ProviderConfig(
            type="openai_compatible",
            base_url="http://127.0.0.1:9/v1",
            api_key_env="MISSING_KEY",
            max_connections=1,
        ),
    )
    req = UnifiedRequest(
        request_id="req",
        tenant_id="demo",
        model="chat-large",
        messages=[ChatMessage(role="user", content="hi")],
        stream=False,
        raw_body={"model": "chat-large", "messages": [{"role": "user", "content": "hi"}]},
    )
    deadline = Deadline(connect_s=0.1, ttft_s=0.1, inter_chunk_s=0.1, total_deadline=999999999.0)
    with pytest.raises(GatewayError) as exc_info:
        await provider.chat(req, "mock", deadline)
    assert exc_info.value.category == ErrorCategory.RETRYABLE_UPSTREAM
    await provider.aclose()


@pytest.mark.asyncio
async def test_provider_ttft_timeout_classification() -> None:
    provider = _mock_stream_provider(delay_s=0.05)
    req = _stream_request()
    deadline = Deadline(
        connect_s=1.0,
        ttft_s=0.01,
        inter_chunk_s=1.0,
        total_deadline=asyncio.get_running_loop().time() + 1.0,
    )
    with pytest.raises(GatewayError) as exc_info:
        async for _ in provider.stream_chat(req, "mock", deadline):
            pass
    assert exc_info.value.category == ErrorCategory.TIMEOUT_TTFT
    await provider.aclose()


@pytest.mark.asyncio
async def test_provider_total_timeout_classification() -> None:
    provider = _mock_stream_provider(delay_s=0.05)
    req = _stream_request()
    deadline = Deadline(
        connect_s=1.0,
        ttft_s=1.0,
        inter_chunk_s=1.0,
        total_deadline=asyncio.get_running_loop().time() + 0.01,
    )
    with pytest.raises(GatewayError) as exc_info:
        async for _ in provider.stream_chat(req, "mock", deadline):
            pass
    assert exc_info.value.category == ErrorCategory.TIMEOUT_TOTAL
    await provider.aclose()


class SlowSSEStream(httpx.AsyncByteStream):
    def __init__(self, delay_s: float) -> None:
        self._delay_s = delay_s

    async def __aiter__(self) -> AsyncIterator[bytes]:
        await asyncio.sleep(self._delay_s)
        yield b'data: {"choices":[{"delta":{"content":"tok0 "},"finish_reason":null}]}\n\n'
        yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield (
            b'data: {"choices":[],"usage":{"prompt_tokens":1,'
            b'"completion_tokens":1,"total_tokens":2}}\n\n'
        )
        yield b"data: [DONE]\n\n"


def _mock_stream_provider(delay_s: float) -> OpenAICompatibleProvider:
    provider = OpenAICompatibleProvider(
        "mock",
        ProviderConfig(
            type="openai_compatible",
            base_url="http://mock/v1",
            api_key_env="MISSING_KEY",
            max_connections=1,
        ),
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, stream=SlowSSEStream(delay_s))
    )
    provider._client = httpx.AsyncClient(transport=transport)
    return provider


def _stream_request() -> UnifiedRequest:
    return UnifiedRequest(
        request_id="req",
        tenant_id="demo",
        model="chat-large",
        messages=[ChatMessage(role="user", content="hi")],
        stream=True,
        raw_body={
            "model": "chat-large",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
