from __future__ import annotations

import pytest

from tidegate.api.errors import _status_for_error
from tidegate.core.errors import ErrorCategory, GatewayError


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        (ErrorCategory.CLIENT_ERROR, 422),
        (ErrorCategory.QUOTA_EXCEEDED, 429),
        (ErrorCategory.RETRYABLE_UPSTREAM, 502),
        (ErrorCategory.RATE_LIMITED_UPSTREAM, 502),
        (ErrorCategory.TIMEOUT_TTFT, 504),
        (ErrorCategory.TIMEOUT_STALL, 504),
        (ErrorCategory.TIMEOUT_TOTAL, 504),
        (ErrorCategory.CONTENT_BLOCKED, 502),
        (ErrorCategory.INTERNAL, 500),
    ],
)
def test_status_for_error_all_categories(category: ErrorCategory, expected: int) -> None:
    """REWORK-M0-6."""
    assert _status_for_error(GatewayError("x", category)) == expected


def test_status_for_error_explicit_status_wins() -> None:
    """REWORK-M0-6."""
    assert _status_for_error(GatewayError("x", ErrorCategory.CLIENT_ERROR, http_status=404)) == 404
