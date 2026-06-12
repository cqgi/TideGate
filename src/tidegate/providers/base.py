from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from tidegate.core.deadline import Deadline
from tidegate.core.models import UnifiedDelta, UnifiedRequest, UnifiedResponse


class Provider(Protocol):
    name: str

    def stream_chat(
        self,
        req: UnifiedRequest,
        upstream_model: str,
        deadline: Deadline,
    ) -> AsyncIterator[UnifiedDelta]: ...

    async def chat(
        self,
        req: UnifiedRequest,
        upstream_model: str,
        deadline: Deadline,
    ) -> UnifiedResponse: ...
