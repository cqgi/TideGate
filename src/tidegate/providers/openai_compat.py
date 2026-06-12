from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from pydantic import ValidationError

from tidegate.config.models import ProviderConfig
from tidegate.core.deadline import Deadline
from tidegate.core.errors import ErrorCategory, GatewayError
from tidegate.core.models import UnifiedDelta, UnifiedRequest, UnifiedResponse, Usage
from tidegate.providers.registry import api_key_from_env, register_provider

_MOCK_DIRECTIVE_BODY_KEY = "__tidegate_mock_directive"


class OpenAICompatibleProvider:
    def __init__(self, name: str, config: ProviderConfig) -> None:
        self.name = name
        self._config = config
        limits = httpx.Limits(max_connections=config.max_connections)
        # SPEC-M1-3: provider instance has an independent pool and ignores proxy env for localhost.
        self._client = httpx.AsyncClient(limits=limits, trust_env=False)
        self._api_key = api_key_from_env(config.api_key_env)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        req: UnifiedRequest,
        upstream_model: str,
        deadline: Deadline,
    ) -> UnifiedResponse:
        # SPEC-M0-4: pass through the OpenAI request body with only model rewritten.
        body, headers = self._upstream_payload(req, upstream_model)
        body["stream"] = False
        try:
            async with asyncio.timeout_at(deadline.total_deadline):
                response = await self._client.post(
                    f"{self._config.base_url}/chat/completions",
                    json=body,
                    headers=headers,
                    timeout=_non_stream_timeout(deadline),
                )
        except httpx.ConnectTimeout as exc:
            # REWORK-M0-3: connect timeout is a retryable connection failure.
            raise GatewayError(
                "upstream connection timed out", ErrorCategory.RETRYABLE_UPSTREAM
            ) from exc
        except httpx.ConnectError as exc:
            raise GatewayError(
                "upstream connection failed", ErrorCategory.RETRYABLE_UPSTREAM
            ) from exc
        except httpx.RemoteProtocolError as exc:
            raise GatewayError("upstream protocol error", ErrorCategory.RETRYABLE_UPSTREAM) from exc
        except httpx.TimeoutException as exc:
            raise GatewayError(
                "upstream total deadline exceeded", ErrorCategory.TIMEOUT_TOTAL
            ) from exc
        except TimeoutError as exc:
            raise GatewayError(
                "upstream total deadline exceeded", ErrorCategory.TIMEOUT_TOTAL
            ) from exc
        _raise_for_status(response)
        payload = response.json()
        return _parse_non_stream_response(payload)

    async def stream_chat(
        self,
        req: UnifiedRequest,
        upstream_model: str,
        deadline: Deadline,
    ) -> AsyncIterator[UnifiedDelta]:
        # SPEC-M0-4: keep httpx stream context around the full generator for cancellation.
        body, headers = self._upstream_payload(req, upstream_model)
        body["stream"] = True
        try:
            async with asyncio.timeout_at(deadline.total_deadline):
                async with self._client.stream(
                    "POST",
                    f"{self._config.base_url}/chat/completions",
                    json=body,
                    headers=headers,
                    timeout=_stream_timeout(deadline),
                ) as response:
                    _raise_for_status(response)
                    first_content = False
                    line_iter = response.aiter_lines()
                    while True:
                        try:
                            if first_content:
                                line = await anext(line_iter)
                            else:
                                # SPEC-M1-1: TTFT is request dispatch until first content delta.
                                async with asyncio.timeout(deadline.ttft_s):
                                    line = await anext(line_iter)
                        except StopAsyncIteration:
                            return
                        except TimeoutError as exc:
                            if deadline.remaining() <= 0:
                                raise GatewayError(
                                    "upstream total deadline exceeded",
                                    ErrorCategory.TIMEOUT_TOTAL,
                                ) from exc
                            raise GatewayError(
                                "upstream time to first token exceeded",
                                ErrorCategory.TIMEOUT_TTFT,
                            ) from exc
                        delta = _parse_sse_line(line)
                        if delta is None:
                            continue
                        if delta.content:
                            first_content = True
                        yield delta
        except httpx.ConnectTimeout as exc:
            # REWORK-M0-3: connect timeout is a retryable connection failure.
            raise GatewayError(
                "upstream connection timed out", ErrorCategory.RETRYABLE_UPSTREAM
            ) from exc
        except httpx.ConnectError as exc:
            raise GatewayError(
                "upstream connection failed", ErrorCategory.RETRYABLE_UPSTREAM
            ) from exc
        except httpx.RemoteProtocolError as exc:
            raise GatewayError("upstream protocol error", ErrorCategory.RETRYABLE_UPSTREAM) from exc
        except httpx.TimeoutException as exc:
            raise GatewayError("upstream stream stalled", ErrorCategory.TIMEOUT_STALL) from exc
        except TimeoutError as exc:
            raise GatewayError(
                "upstream total deadline exceeded", ErrorCategory.TIMEOUT_TOTAL
            ) from exc

    def _upstream_payload(
        self,
        req: UnifiedRequest,
        upstream_model: str,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        body = dict(req.raw_body)
        body["model"] = upstream_model
        headers = {"Authorization": f"Bearer {self._api_key}"}
        # DECISION: M0 forwards mock directives for deterministic integration tests.
        mock_directive = body.pop(_MOCK_DIRECTIVE_BODY_KEY, None)
        if isinstance(mock_directive, str):
            headers["x-mock-directive"] = mock_directive
        return body, headers


def _stream_timeout(deadline: Deadline) -> httpx.Timeout:
    return httpx.Timeout(
        connect=deadline.connect_s,
        read=deadline.inter_chunk_s,
        write=deadline.connect_s,
        pool=deadline.connect_s,
    )


def _non_stream_timeout(deadline: Deadline) -> httpx.Timeout:
    # DECISION: REWORK-M1-3 leaves non-stream response reads governed by the total
    # deadline, because a valid long answer arrives as one completed response body.
    return httpx.Timeout(
        connect=deadline.connect_s,
        read=None,
        write=deadline.connect_s,
        pool=deadline.connect_s,
    )


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    retry_after = response.headers.get("Retry-After")
    retry_after_s = _parse_retry_after(retry_after)
    if response.status_code == 429:
        raise GatewayError(
            "upstream rate limited",
            ErrorCategory.RATE_LIMITED_UPSTREAM,
            retry_after_s=retry_after_s,
        )
    if response.status_code >= 500:
        raise GatewayError("upstream error", ErrorCategory.RETRYABLE_UPSTREAM)
    raise GatewayError("upstream rejected request", ErrorCategory.CLIENT_ERROR, http_status=422)


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            # REWORK-M0-3: invalid Retry-After values are ignored, not leaked as ValueError.
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return max(0.0, (parsed - datetime.now(UTC)).total_seconds())


def _parse_non_stream_response(payload: dict[str, Any]) -> UnifiedResponse:
    try:
        choice = payload["choices"][0]
        message = choice["message"]
        usage = Usage.model_validate(payload["usage"])
        return UnifiedResponse(
            content=str(message.get("content") or ""),
            finish_reason=str(choice.get("finish_reason") or "stop"),
            usage=usage,
            model=str(payload["model"]),
            mean_logprob=_mean_logprob(choice),
        )
    except (KeyError, IndexError, TypeError, ValidationError) as exc:
        raise GatewayError("invalid upstream response", ErrorCategory.RETRYABLE_UPSTREAM) from exc


def _mean_logprob(choice: dict[str, Any]) -> float | None:
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return None
    content = logprobs.get("content")
    if not isinstance(content, list) or not content:
        return None
    values: list[float] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        value = item.get("logprob")
        if isinstance(value, int | float):
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def _parse_sse_line(line: str) -> UnifiedDelta | None:
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    data = line.removeprefix("data:").strip()
    if data == "[DONE]":
        return None
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        # REWORK-M0-3: bad upstream SSE JSON is an upstream protocol error.
        raise GatewayError("invalid upstream SSE", ErrorCategory.RETRYABLE_UPSTREAM) from exc
    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    if "usage" in payload and payload["usage"] is not None:
        return UnifiedDelta(
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            usage=Usage.model_validate(payload["usage"]),
            raw=payload,
        )

    if not isinstance(choice, dict):
        return None
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        delta = {}
    content = delta.get("content")
    return UnifiedDelta(
        content=content if isinstance(content, str) else None,
        finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        raw=payload,
    )


@register_provider("openai_compatible")
def build_openai_compatible(
    name: str,
    config: ProviderConfig,
) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(name, config)
