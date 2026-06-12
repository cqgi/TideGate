from __future__ import annotations

import logging

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars


def configure_logging() -> None:
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )


def bind_request_id(request_id: str) -> None:
    clear_contextvars()
    bind_contextvars(request_id=request_id)
