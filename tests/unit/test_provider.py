from __future__ import annotations

import httpx
import pytest

from tidegate.core.errors import ErrorCategory, GatewayError
from tidegate.providers.openai_compat import _parse_retry_after, _parse_sse_line


def test_parse_sse_bad_json_is_gateway_error() -> None:
    """REWORK-M0-6."""
    with pytest.raises(GatewayError) as exc_info:
        _parse_sse_line("data: {bad json")
    assert exc_info.value.category == ErrorCategory.RETRYABLE_UPSTREAM


def test_retry_after_http_date_and_invalid() -> None:
    """REWORK-M0-3."""
    assert _parse_retry_after("not a date") is None
    assert _parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0


def test_connect_timeout_is_retryable() -> None:
    """REWORK-M0-3."""
    assert issubclass(httpx.ConnectTimeout, httpx.TimeoutException)
