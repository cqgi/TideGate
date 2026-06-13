from __future__ import annotations

from pathlib import Path

from tidegate.cache.gates import can_store, l2_context_allowed, read_decision
from tidegate.config.loader import load_config
from tidegate.core.models import ChatMessage, UnifiedRequest, UnifiedResponse, Usage


def test_read_gate_bypasses_volatile_and_tools() -> None:
    settings = load_config(Path("tests/fixtures/gateway-test.yaml"))
    tenant = settings.tenants[0]

    assert read_decision(_req("今天北京天气"), tenant, settings).bypass_reason == "volatile"
    assert read_decision(_req("hi", has_tools=True), tenant, settings).bypass_reason == "tools"


def test_l2_requires_single_user_turn() -> None:
    assert l2_context_allowed(_req("单轮"))
    assert not l2_context_allowed(
        _req(
            "多轮",
            messages=[
                ChatMessage(role="user", content="a"),
                ChatMessage(role="user", content="b"),
            ],
        )
    )


def test_store_gate_rejects_degraded_and_bad_content() -> None:
    settings = load_config(Path("tests/fixtures/gateway-test.yaml"))
    tenant = settings.tenants[0]

    assert can_store(_req("hi"), tenant, _resp("ok"), settings, degraded=False) == (True, False)
    assert can_store(_req("hi"), tenant, _resp("ok"), settings, degraded=True) == (False, False)
    assert can_store(_req("hi"), tenant, _resp("抱歉无法回答"), settings, degraded=False) == (
        False,
        False,
    )


def _req(
    content: str,
    *,
    has_tools: bool = False,
    messages: list[ChatMessage] | None = None,
) -> UnifiedRequest:
    return UnifiedRequest(
        request_id="req",
        tenant_id="demo",
        model="chat-large",
        messages=messages or [ChatMessage(role="user", content=content)],
        stream=False,
        temperature=0.1,
        has_tools=has_tools,
        raw_body={},
    )


def _resp(content: str) -> UnifiedResponse:
    return UnifiedResponse(
        content=content,
        finish_reason="stop",
        model="mock",
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
