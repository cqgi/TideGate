from __future__ import annotations

import hashlib
import json
from typing import Any

from tidegate.core.models import UnifiedRequest

_CANONICAL_FIELDS = {
    "model",
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "prompt_version",
}


def canonical_form(req: UnifiedRequest) -> dict[str, object]:
    canonical: dict[str, object] = {
        "model": req.model,
        "messages": [
            {"role": message.role, "content": (message.content or "").strip()}
            for message in req.messages
        ],
        "temperature": None if req.temperature is None else round(req.temperature, 2),
        "top_p": None if req.top_p is None else round(req.top_p, 2),
        "max_tokens": req.max_tokens,
        "stop": req.stop,
        "prompt_version": req.prompt_version,
    }
    assert set(canonical) == _CANONICAL_FIELDS
    return canonical


def l1_digest(req: UnifiedRequest) -> str:
    payload = json.dumps(canonical_form(req), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def semantic_text(req: UnifiedRequest) -> str:
    return "\n".join((message.content or "").strip() for message in req.messages)


def canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
