from __future__ import annotations

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from tidegate.core.errors import ErrorCategory, GatewayError

_ERROR_TYPE_BY_STATUS = {
    401: "authentication_error",
    404: "invalid_request_error",
    422: "invalid_request_error",
    429: "rate_limit_error",
    502: "upstream_error",
    503: "internal_error",
    504: "timeout_error",
    500: "internal_error",
}


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(GatewayError)
    async def gateway_error_handler(request: Request, exc: GatewayError) -> JSONResponse:
        return gateway_error_response(request, exc)

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        del exc
        return gateway_error_response(
            request,
            GatewayError(
                "request validation failed",
                ErrorCategory.CLIENT_ERROR,
                http_status=422,
            ),
        )

    @app.exception_handler(ValidationError)
    async def validation_handler(request: Request, exc: ValidationError) -> JSONResponse:
        del exc
        return gateway_error_response(
            request,
            GatewayError(
                "request validation failed",
                ErrorCategory.CLIENT_ERROR,
                http_status=422,
            ),
        )

    @app.exception_handler(Exception)
    async def unknown_error_handler(request: Request, exc: Exception) -> JSONResponse:
        structlog.get_logger().exception("unhandled_error", path=request.url.path)
        return gateway_error_response(
            request,
            GatewayError(
                "internal server error",
                ErrorCategory.INTERNAL,
                http_status=500,
            ),
        )


def gateway_error_response(request: Request, exc: GatewayError) -> JSONResponse:
    status = _status_for_error(exc)
    error_type = _ERROR_TYPE_BY_STATUS[status]
    code = exc.code
    headers: dict[str, str] = {}
    if status == 429:
        retry_after = exc.retry_after_s if exc.retry_after_s is not None else 1.0
        headers["Retry-After"] = str(int(retry_after + 0.999))
        code = code or "rate_limited"
    body = {"error": {"message": exc.message, "type": error_type, "code": code}}
    request_id = getattr(request.state, "request_id", None)
    if isinstance(request_id, str):
        headers["X-Request-Id"] = request_id
    headers.setdefault("X-TideGate-Cache", "bypass")
    headers.setdefault("X-TideGate-Route", "none")
    return JSONResponse(body, status_code=status, headers=headers)


def _status_for_error(exc: GatewayError) -> int:
    if exc.http_status is not None:
        return exc.http_status
    match exc.category:
        case ErrorCategory.CLIENT_ERROR:
            return 422
        case ErrorCategory.QUOTA_EXCEEDED:
            return 429
        case ErrorCategory.RETRYABLE_UPSTREAM | ErrorCategory.RATE_LIMITED_UPSTREAM:
            return 502
        case ErrorCategory.TIMEOUT_TTFT | ErrorCategory.TIMEOUT_STALL | ErrorCategory.TIMEOUT_TOTAL:
            return 504
        case ErrorCategory.CONTENT_BLOCKED:
            return 502
        case ErrorCategory.INTERNAL:
            return 500
