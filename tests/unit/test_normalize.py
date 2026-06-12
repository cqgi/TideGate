from __future__ import annotations

from tidegate.cache.normalize import canonical_form, l1_digest
from tidegate.core.models import ChatMessage, UnifiedRequest


def test_l1_key_is_stable_for_whitespace_and_dict_order() -> None:
    """SPEC-M4-1."""
    req_a = _req(content="  怎么退款  ", raw_body={"b": 2, "a": 1})
    req_b = _req(content="怎么退款", raw_body={"a": 1, "b": 2})

    assert l1_digest(req_a) == l1_digest(req_b)


def test_canonical_field_set_is_frozen() -> None:
    """SPEC-M4-1."""
    req = _req(content="hi")

    assert set(canonical_form(req)) == {
        "model",
        "messages",
        "temperature",
        "top_p",
        "max_tokens",
        "stop",
        "prompt_version",
    }


def _req(content: str, raw_body: dict[str, object] | None = None) -> UnifiedRequest:
    return UnifiedRequest(
        request_id="req",
        tenant_id="demo",
        model="chat-large",
        messages=[ChatMessage(role="user", content=content)],
        stream=False,
        temperature=0.123,
        top_p=0.987,
        max_tokens=16,
        stop=["END"],
        prompt_version="v1",
        raw_body=raw_body or {},
    )
