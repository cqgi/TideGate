from __future__ import annotations

from contextlib import AbstractContextManager

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Span
from opentelemetry.util.types import Attributes

from tidegate.config.models import OTelConfig

_TRACER_NAME = "tidegate"


def configure_otel(config: OTelConfig) -> None:
    if config.exporter == "none":
        return
    provider = TracerProvider(resource=Resource.create({"service.name": "tidegate"}))
    if config.exporter == "console":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif config.exporter == "otlp":
        exporter = OTLPSpanExporter(endpoint=config.endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


def start_span(name: str, attributes: Attributes = None) -> AbstractContextManager[Span]:
    return trace.get_tracer(_TRACER_NAME).start_as_current_span(name, attributes=attributes)


def current_trace_id() -> str | None:
    span = trace.get_current_span()
    context = span.get_span_context()
    if not context.is_valid:
        return None
    return format(context.trace_id, "032x")
