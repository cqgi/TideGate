from __future__ import annotations

import asyncio

from tidegate.obs.metrics import Metrics


async def probe_loop_lag(metrics: Metrics, interval_s: float) -> None:
    # Baseline loop-lag probe for event-loop health.
    loop = asyncio.get_running_loop()
    while True:
        expected = loop.time() + interval_s
        await asyncio.sleep(interval_s)
        metrics.loop_lag.set(max(0.0, loop.time() - expected))
