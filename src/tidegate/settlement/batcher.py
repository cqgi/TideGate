from __future__ import annotations

import asyncio
from dataclasses import dataclass

import asyncpg
import structlog

from tidegate.config.models import SettlementConfig
from tidegate.core.models import Usage
from tidegate.obs.metrics import Metrics


@dataclass(frozen=True)
class LedgerRecord:
    request_id: str
    tenant_id: str
    model: str
    provider: str
    upstream_model: str
    usage: Usage
    cost_microusd: int
    cache_status: str
    route_path: str
    degraded: str | None
    outcome: str


class LedgerBatcher:
    _drain_timeout_s = 10.0

    def __init__(
        self,
        pool: asyncpg.Pool | None,
        config: SettlementConfig,
        metrics: Metrics,
    ) -> None:
        self._pool = pool
        self._config = config
        self._metrics = metrics
        self._queue: asyncio.Queue[LedgerRecord] = asyncio.Queue(maxsize=config.queue_max)
        self._closed = False
        self._drain_lock = asyncio.Lock()

    def enqueue(self, record: LedgerRecord) -> None:
        if self._pool is None or self._closed:
            return
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            self._metrics.ledger_dropped.inc()
            structlog.get_logger().warning("ledger_queue_full", tenant=record.tenant_id)

    async def enqueue_and_flush(self, record: LedgerRecord) -> None:
        if self._pool is None or self._closed:
            return
        await self._write_with_retry([record])

    async def run(self) -> None:
        if self._pool is None:
            return
        batch: list[LedgerRecord] = []
        try:
            while True:
                batch = [await self._queue.get()]
                deadline = asyncio.get_running_loop().time() + self._config.batch_interval_ms / 1000
                while len(batch) < self._config.batch_size:
                    timeout = max(0.0, deadline - asyncio.get_running_loop().time())
                    if timeout == 0:
                        break
                    try:
                        batch.append(await asyncio.wait_for(self._queue.get(), timeout=timeout))
                    except TimeoutError:
                        break
                await self._write_with_retry(batch)
                for _ in batch:
                    self._queue.task_done()
                batch = []
        except asyncio.CancelledError:
            if batch:
                await self._write_with_retry(batch)
                for _ in batch:
                    self._queue.task_done()
            await self.drain()
            raise

    async def drain(self) -> None:
        async with self._drain_lock:
            if self._pool is None:
                return
            batch: list[LedgerRecord] = []
            deadline = asyncio.get_running_loop().time() + self._drain_timeout_s
            while not self._queue.empty():
                batch.append(self._queue.get_nowait())
                if len(batch) >= self._config.batch_size:
                    await self._write_with_retry(batch, deadline_s=deadline)
                    for _ in batch:
                        self._queue.task_done()
                    batch = []
            if batch:
                await self._write_with_retry(batch, deadline_s=deadline)
                for _ in batch:
                    self._queue.task_done()

    def close(self) -> None:
        self._closed = True

    async def _write_with_retry(
        self,
        batch: list[LedgerRecord],
        *,
        deadline_s: float | None = None,
    ) -> None:
        if self._pool is None or not batch:
            return
        delay = 0.05
        while True:
            try:
                await self._write(batch)
                return
            except (asyncpg.PostgresError, OSError, ConnectionError) as exc:
                if deadline_s is not None and asyncio.get_running_loop().time() >= deadline_s:
                    self._metrics.ledger_dropped.inc(len(batch))
                    structlog.get_logger().warning(
                        "ledger_drain_deadline_exceeded",
                        records=len(batch),
                        error=str(exc),
                    )
                    return
                structlog.get_logger().warning("ledger_write_failed", error=str(exc))
                await asyncio.sleep(delay)
                delay = min(1.0, delay * 2)

    async def _write(self, batch: list[LedgerRecord]) -> None:
        assert self._pool is not None
        rows = [
            (
                record.request_id,
                record.tenant_id,
                record.model,
                record.provider,
                record.upstream_model,
                record.usage.prompt_tokens,
                record.usage.completion_tokens,
                record.cost_microusd,
                record.cache_status,
                record.route_path,
                record.degraded,
                record.outcome,
            )
            for record in batch
        ]
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO usage_ledger (
                  request_id, tenant_id, model, provider, upstream_model,
                  prompt_tokens, completion_tokens, cost_microusd,
                  cache_status, route_path, degraded, outcome
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                ON CONFLICT (request_id) DO NOTHING
                """,
                rows,
            )
