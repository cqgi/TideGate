from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass(frozen=True)
class Deadline:
    connect_s: float
    ttft_s: float
    inter_chunk_s: float
    total_deadline: float

    def remaining(self) -> float:
        return max(0.0, self.total_deadline - asyncio.get_running_loop().time())
