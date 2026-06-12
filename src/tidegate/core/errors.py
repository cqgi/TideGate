from __future__ import annotations

from enum import StrEnum


class ErrorCategory(StrEnum):
    RETRYABLE_UPSTREAM = "retryable_upstream"
    RATE_LIMITED_UPSTREAM = "rate_limited"
    CONTENT_BLOCKED = "content_blocked"
    TIMEOUT_TTFT = "timeout_ttft"
    TIMEOUT_STALL = "timeout_stall"
    TIMEOUT_TOTAL = "timeout_total"
    QUOTA_EXCEEDED = "quota_exceeded"
    CLIENT_ERROR = "client_error"
    INTERNAL = "internal"


class GatewayError(Exception):
    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        *,
        retry_after_s: float | None = None,
        code: str | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.retry_after_s = retry_after_s
        self.message = message
        self.code = code
        self.http_status = http_status
