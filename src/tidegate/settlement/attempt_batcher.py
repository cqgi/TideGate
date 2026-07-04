from __future__ import annotations

import asyncio
from dataclasses import dataclass

import asyncpg
import structlog

from tidegate.config.models import SettlementConfig
from tidegate.obs.metrics import Metrics


@dataclass(frozen=True)
class AttemptLedgerRecord:
    request_id: str
    attempt_seq: int
    tenant_id: str
    kind: str
    outcome: str
    provider: str | None
    upstream_model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    usage_source: str
    cost_tenant_microusd: int
    cost_platform_microusd: int
    bearer_reason: str | None
    error_category: str | None = None


class AttemptLedgerBatcher:
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
        self._queue: asyncio.Queue[AttemptLedgerRecord] = asyncio.Queue(maxsize=config.queue_max)
        self._closed = False
        self._drain_lock = asyncio.Lock()

    def enqueue(self, record: AttemptLedgerRecord) -> None:
        if self._pool is None or self._closed:
            return
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            self._metrics.ledger_dropped.inc()
            structlog.get_logger().warning(
                "attempt_ledger_queue_full",
                tenant=record.tenant_id,
                request_id=record.request_id,
            )

    async def enqueue_and_flush(self, record: AttemptLedgerRecord) -> None:
        if self._pool is None or self._closed:
            return
        await self._write_with_retry([record])

    async def run(self) -> None:
        if self._pool is None:
            return
        batch: list[AttemptLedgerRecord] = []
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
            batch: list[AttemptLedgerRecord] = []
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
        batch: list[AttemptLedgerRecord],
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
                        "attempt_ledger_drain_deadline_exceeded",
                        records=len(batch),
                        error=str(exc),
                    )
                    return
                structlog.get_logger().warning("attempt_ledger_write_failed", error=str(exc))
                await asyncio.sleep(delay)
                delay = min(1.0, delay * 2)

    async def _write(self, batch: list[AttemptLedgerRecord]) -> None:
        assert self._pool is not None
        rows = [
            (
                record.request_id,
                record.attempt_seq,
                record.tenant_id,
                record.kind,
                record.outcome,
                record.provider,
                record.upstream_model,
                record.prompt_tokens,
                record.completion_tokens,
                record.usage_source,
                record.cost_tenant_microusd,
                record.cost_platform_microusd,
                record.bearer_reason,
                record.error_category,
            )
            for record in batch
        ]
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO attempt_ledger (
                  request_id, attempt_seq, tenant_id, kind, outcome,
                  provider, upstream_model, prompt_tokens, completion_tokens,
                  usage_source, cost_tenant_microusd, cost_platform_microusd,
                  bearer_reason, error_category
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                ON CONFLICT (request_id, attempt_seq) DO NOTHING
                """,
                rows,
            )
