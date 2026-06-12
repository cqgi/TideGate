from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import asyncpg
import httpx
import pytest
import yaml

from tests.integration.conftest import (
    API_KEY,
    MOCK_A_URL,
    MOCK_B_URL,
    reset_mock,
    wait_ready,
)
from tidegate.config.models import SettlementConfig
from tidegate.core.models import Usage
from tidegate.obs.metrics import Metrics
from tidegate.settlement import LedgerBatcher, LedgerRecord

PG_DSN = "postgresql://tidegate:tidegate@127.0.0.1:5432/tidegate"


@pytest.mark.integration
def test_ledger_writes_concurrent_requests(
    mock_a_proc: subprocess.Popen[str],
    mock_b_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    """SPEC-M5-3."""
    del mock_a_proc, mock_b_proc
    reset_mock(MOCK_A_URL)
    reset_mock(MOCK_B_URL)
    port = 8055
    proc = _start_gateway(_ledger_config(tmp_path, port), port)

    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            for index in range(20):
                response = _chat(client, port, f"ledger {index}")
                assert response.status_code == 200, response.text
            count = _wait_for_ledger_count(20)
    finally:
        _stop_gateway(proc)

    assert count == 20


@pytest.mark.integration
def test_ledger_survives_gateway_sigterm(
    mock_a_proc: subprocess.Popen[str],
    mock_b_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    """SPEC-M5-3."""
    del mock_a_proc, mock_b_proc
    reset_mock(MOCK_A_URL)
    reset_mock(MOCK_B_URL)
    port = 8056
    proc = _start_gateway(_ledger_config(tmp_path, port), port)

    completed = 0
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            for index in range(10):
                response = _chat(client, port, f"ledger shutdown {index}")
                assert response.status_code == 200, response.text
                completed += 1
        proc.terminate()
        proc.wait(timeout=10)
        count = _wait_for_ledger_count(completed)
    finally:
        if proc.poll() is None:
            _stop_gateway(proc)

    assert count == completed


@pytest.mark.integration
def test_ledger_batcher_is_idempotent_for_duplicate_request_id() -> None:
    """SPEC-M5-3."""
    _truncate_ledger()

    async def run() -> int:
        pool = await asyncpg.create_pool(PG_DSN)
        assert pool is not None
        try:
            batcher = LedgerBatcher(
                pool,
                SettlementConfig(batch_size=10, batch_interval_ms=50, queue_max=100),
                Metrics.create(),
            )
            record = _ledger_record("duplicate-request")
            await batcher._write([record, record])
            async with pool.acquire() as conn:
                return int(await conn.fetchval("SELECT count(*) FROM usage_ledger"))
        finally:
            await pool.close()

    assert asyncio.run(run()) == 1


@pytest.mark.integration
def test_ledger_buffers_while_postgres_restarts(
    mock_a_proc: subprocess.Popen[str],
    mock_b_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    """SPEC-M5-3."""
    del mock_a_proc, mock_b_proc
    reset_mock(MOCK_A_URL)
    reset_mock(MOCK_B_URL)
    port = 8057
    proc = _start_gateway(_ledger_config(tmp_path, port), port)

    try:
        subprocess.run(
            ["docker", "compose", "-f", "deploy/docker-compose.yml", "stop", "postgres"],
            check=True,
            capture_output=True,
            text=True,
        )
        with httpx.Client(timeout=5, trust_env=False) as client:
            for index in range(3):
                response = _chat(client, port, f"ledger pg restart {index}")
                assert response.status_code == 200, response.text
        subprocess.run(
            ["docker", "compose", "-f", "deploy/docker-compose.yml", "start", "postgres"],
            check=True,
            capture_output=True,
            text=True,
        )
        count = _wait_for_ledger_count(3, timeout_s=15)
    finally:
        subprocess.run(
            ["docker", "compose", "-f", "deploy/docker-compose.yml", "start", "postgres"],
            capture_output=True,
            text=True,
        )
        if proc.poll() is None:
            _stop_gateway(proc)

    assert count == 3


def _chat(client: httpx.Client, port: int, content: str) -> httpx.Response:
    return client.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": "chat-large", "messages": [{"role": "user", "content": content}]},
    )


def _ledger_config(tmp_path: Path, port: int) -> Path:
    raw = yaml.safe_load(Path("tests/fixtures/gateway-test.yaml").read_text(encoding="utf-8"))
    raw["server"]["port"] = port
    raw["settlement"] = {"batch_size": 5, "batch_interval_ms": 50, "queue_max": 1000}
    raw["tenants"][0]["cache"] = {"l1": False, "l2": False}
    path = tmp_path / f"gateway-m5-ledger-{port}.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def _start_gateway(config_path: Path, port: int) -> subprocess.Popen[str]:
    env = {
        **os.environ,
        "TIDEGATE_ADMIN_TOKEN": "dev-admin",
        "MOCK_A_KEY": "mock-key",
        "MOCK_B_KEY": "mock-key",
        "TIDEGATE_PG_DSN": PG_DSN,
        "PYTHONPATH": f"{Path.cwd() / 'src'}:{Path.cwd()}",
    }
    _truncate_ledger()
    proc = subprocess.Popen(
        [sys.executable, "-m", "tidegate", "--config", str(config_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    wait_ready(f"http://127.0.0.1:{port}/healthz", proc)
    return proc


def _truncate_ledger() -> None:
    async def run() -> None:
        conn = await _connect_pg()
        try:
            ddl = await asyncio.to_thread(
                Path("deploy/sql/001_usage_ledger.sql").read_text,
                encoding="utf-8",
            )
            await conn.execute(ddl)
            await conn.execute("TRUNCATE usage_ledger")
        finally:
            await conn.close()

    asyncio.run(run())


def _ledger_record(request_id: str) -> LedgerRecord:
    usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return LedgerRecord(
        request_id=request_id,
        tenant_id="demo",
        model="chat-large",
        provider="mock-a",
        upstream_model="mock-gpt-large",
        usage=usage,
        cost_microusd=1,
        cache_status="miss",
        route_path="mock-a/mock-gpt-large",
        degraded=None,
        outcome="ok",
    )


def _wait_for_ledger_count(expected: int, *, timeout_s: float = 5) -> int:
    async def run() -> int:
        deadline = time.monotonic() + timeout_s
        conn = await _connect_pg()
        try:
            count = 0
            while time.monotonic() < deadline:
                count = int(await conn.fetchval("SELECT count(*) FROM usage_ledger"))
                if count >= expected:
                    return count
                await asyncio.sleep(0.05)
            return count
        finally:
            await conn.close()

    return asyncio.run(run())


async def _connect_pg() -> asyncpg.Connection:
    deadline = time.monotonic() + 10
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return await asyncpg.connect(PG_DSN)
        except (OSError, asyncpg.PostgresError) as exc:
            last_error = exc
            await asyncio.sleep(0.1)
    raise RuntimeError(f"postgres not ready: {last_error}")


def _stop_gateway(proc: subprocess.Popen[str]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
