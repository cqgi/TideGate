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
import redis
import yaml

from tests.integration.conftest import (
    API_KEY,
    MOCK_A_URL,
    MOCK_B_URL,
    reset_mock,
    wait_ready,
)

PG_DSN = os.environ.get(
    "TIDEGATE_TEST_PG_DSN",
    "postgresql://tidegate:tidegate@127.0.0.1:5432/tidegate",
)


@pytest.mark.integration
def test_m6_hedge_loser_attempt_is_platform_costed(
    redis_stack_proc: None,
    mock_a_proc: subprocess.Popen[str],
    mock_b_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    """SPEC-M6-2: a hedged loser is an estimated platform-cost attempt row."""

    del redis_stack_proc, mock_a_proc, mock_b_proc
    reset_mock(MOCK_A_URL)
    reset_mock(MOCK_B_URL)
    _set_behavior(MOCK_A_URL, {"ttft_ms": 200, "output_tokens": 3})
    _set_behavior(MOCK_B_URL, {"ttft_ms": 20, "output_tokens": 3})
    port = 8061
    proc = _start_gateway(_hedge_config(tmp_path, port), port)

    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            response = _stream_chat(client, port)
            metrics = client.get(f"http://127.0.0.1:{port}/metrics").text
        rows = _wait_for_attempt_rows(response.headers["x-request-id"], expected=2)
    finally:
        _stop_gateway(proc)

    assert response.status_code == 200, response.text
    assert "tok0" in response.text
    assert len(rows) == 2
    loser = _single_row(rows, outcome="hedge_loser")
    winner = _single_row(rows, outcome="delivered")
    assert {row["attempt_seq"] for row in rows} == {0, 1}
    assert loser["kind"] in {"primary", "hedge"}
    assert loser["provider"] == "mock-a"
    assert loser["usage_source"] == "estimated"
    assert _int_value(loser, "prompt_tokens") > 0
    assert loser["completion_tokens"] == 0
    assert loser["cost_tenant_microusd"] == 0
    assert _int_value(loser, "cost_platform_microusd") > 0
    assert loser["bearer_reason"] == "hedge_loser"
    assert winner["kind"] in {"primary", "hedge"}
    assert winner["kind"] != loser["kind"]
    assert winner["provider"] == "mock-b"
    assert winner["usage_source"] == "actual"
    assert _int_value(winner, "cost_tenant_microusd") > 0
    assert winner["cost_platform_microusd"] == 0
    assert 'tidegate_platform_cost_microusd_total{reason="hedge_loser",tenant="demo"}' in metrics


def _stream_chat(client: httpx.Client, port: int) -> httpx.Response:
    return client.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": "chat-large",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "hedge stream"}],
        },
    )


def _single_row(rows: list[dict[str, object]], *, outcome: str) -> dict[str, object]:
    matches = [row for row in rows if row["outcome"] == outcome]
    assert len(matches) == 1
    return matches[0]


def _int_value(row: dict[str, object], key: str) -> int:
    value = row[key]
    assert isinstance(value, int)
    return value


def _set_behavior(url: str, behavior: dict[str, object]) -> None:
    with httpx.Client(timeout=2, trust_env=False) as client:
        response = client.post(f"{url}/__behavior", json=behavior)
    assert response.status_code == 200, response.text


def _hedge_config(tmp_path: Path, port: int) -> Path:
    raw = yaml.safe_load(Path("tests/fixtures/gateway-test.yaml").read_text(encoding="utf-8"))
    raw["server"]["port"] = port
    raw["settlement"] = {
        "batch_size": 5,
        "batch_interval_ms": 50,
        "queue_max": 1000,
        "drain_timeout_s": 10.0,
    }
    raw["routing"]["p2c_weights"] = {"ttft": 0.0, "error_rate": 0.0, "inflight": 0.0, "price": 1.0}
    raw["policies"]["default"]["hedging"] = {
        "enabled": True,
        "trigger_quantile": 0.95,
        "trigger_floor_s": 0.01,
        "max_hedge_ratio": 1.0,
    }
    raw["model_groups"]["chat-large"]["deployments"] = [
        {
            "provider": "mock-a",
            "upstream_model": "mock-gpt-large",
            "weight": 1,
            "price_per_1k_input_usd": 0.001,
            "price_per_1k_output_usd": 0.002,
            "supports_logprobs": True,
        },
        {
            "provider": "mock-b",
            "upstream_model": "mock-gpt-large",
            "weight": 1,
            "price_per_1k_input_usd": 0.01,
            "price_per_1k_output_usd": 0.02,
            "supports_logprobs": True,
        },
    ]
    raw["tenants"][0]["cache"] = {"l1": False, "l2": False}
    path = tmp_path / f"gateway-m6-attempt-ledger-{port}.yaml"
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
    _truncate_ledgers()
    redis.Redis.from_url("redis://127.0.0.1:6379/0").flushdb()
    proc = subprocess.Popen(
        [sys.executable, "-m", "tidegate", "--config", str(config_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    wait_ready(f"http://127.0.0.1:{port}/healthz", proc)
    return proc


def _stop_gateway(proc: subprocess.Popen[str]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _truncate_ledgers() -> None:
    ddl = (
        Path("deploy/sql/001_usage_ledger.sql").read_text(encoding="utf-8")
        + "\n"
        + Path("deploy/sql/002_attempt_ledger.sql").read_text(encoding="utf-8")
    )

    async def run() -> None:
        conn = await asyncpg.connect(PG_DSN)
        try:
            await conn.execute(ddl)
            await conn.execute("TRUNCATE usage_ledger, attempt_ledger")
        finally:
            await conn.close()

    asyncio.run(run())


def _wait_for_attempt_rows(
    request_id: str,
    *,
    expected: int,
    timeout_s: float = 5,
) -> list[dict[str, object]]:
    async def run() -> list[dict[str, object]]:
        deadline = time.monotonic() + timeout_s
        conn = await asyncpg.connect(PG_DSN)
        try:
            rows: list[dict[str, object]] = []
            while time.monotonic() < deadline:
                rows = [
                    dict(row)
                    for row in await conn.fetch(
                        """
                        SELECT request_id, attempt_seq, tenant_id, kind, outcome,
                               provider, upstream_model, prompt_tokens, completion_tokens,
                               usage_source, cost_tenant_microusd, cost_platform_microusd,
                               bearer_reason, error_category
                        FROM attempt_ledger
                        WHERE request_id = $1
                        ORDER BY attempt_seq
                        """,
                        request_id,
                    )
                ]
                if len(rows) >= expected:
                    return rows
                await asyncio.sleep(0.05)
            return rows
        finally:
            await conn.close()

    return asyncio.run(run())
