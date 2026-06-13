from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import yaml

from tests.integration.conftest import (
    API_KEY,
    MOCK_A_URL,
    MOCK_B_URL,
    reset_mock,
    stats,
    wait_ready,
)


@pytest.mark.integration
def test_hedge_stream_winner_aborts_loser(
    mock_a_proc: subprocess.Popen[str],
    mock_b_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    del mock_a_proc, mock_b_proc
    reset_mock(MOCK_A_URL)
    reset_mock(MOCK_B_URL)
    _set_behavior(MOCK_A_URL, {"ttft_ms": 200, "output_tokens": 3})
    _set_behavior(MOCK_B_URL, {"ttft_ms": 20, "output_tokens": 3})
    port = 8053
    proc = _start_gateway(_hedge_config(tmp_path, port), port)

    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            started = time.monotonic()
            response = _stream_chat(client, port)
            elapsed = time.monotonic() - started
            mock_a_stats = stats(MOCK_A_URL)
            mock_b_stats = stats(MOCK_B_URL)
            metrics = client.get(f"http://127.0.0.1:{port}/metrics").text
    finally:
        _stop_gateway(proc)

    assert response.status_code == 200, response.text
    assert "tok0" in response.text
    assert elapsed < 1.0
    assert mock_a_stats["aborted"] == 1
    assert mock_b_stats["completed"] == 1
    assert 'tidegate_hedge_total{outcome="won"} 1.0' in metrics


@pytest.mark.integration
def test_hedge_budget_skips_after_limit(
    mock_a_proc: subprocess.Popen[str],
    mock_b_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    del mock_a_proc, mock_b_proc
    reset_mock(MOCK_A_URL)
    reset_mock(MOCK_B_URL)
    _set_behavior(MOCK_A_URL, {"ttft_ms": 80, "output_tokens": 1})
    _set_behavior(MOCK_B_URL, {"ttft_ms": 10, "output_tokens": 1})
    port = 8054
    proc = _start_gateway(_hedge_config(tmp_path, port, max_hedge_ratio=0.01), port)

    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            for index in range(5):
                response = _stream_chat(client, port, content=f"hedge budget {index}")
                assert response.status_code == 200, response.text
            metrics = client.get(f"http://127.0.0.1:{port}/metrics").text
    finally:
        _stop_gateway(proc)

    assert 'tidegate_hedge_total{outcome="skipped_budget"}' in metrics


def _stream_chat(
    client: httpx.Client,
    port: int,
    *,
    content: str = "hedge stream",
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    merged_headers = {"Authorization": f"Bearer {API_KEY}"}
    if headers is not None:
        merged_headers.update(headers)
    return client.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        headers=merged_headers,
        json={
            "model": "chat-large",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": content}],
        },
    )


def _set_behavior(url: str, behavior: dict[str, object]) -> None:
    with httpx.Client(timeout=2, trust_env=False) as client:
        response = client.post(f"{url}/__behavior", json=behavior)
    assert response.status_code == 200, response.text


def _hedge_config(tmp_path: Path, port: int, *, max_hedge_ratio: float = 1.0) -> Path:
    raw = yaml.safe_load(Path("tests/fixtures/gateway-test.yaml").read_text(encoding="utf-8"))
    raw["server"]["port"] = port
    raw["routing"]["p2c_weights"] = {"ttft": 0.0, "error_rate": 0.0, "inflight": 0.0, "price": 1.0}
    raw["policies"]["default"]["hedging"] = {
        "enabled": True,
        "trigger_quantile": 0.95,
        "trigger_floor_s": 0.01,
        "max_hedge_ratio": max_hedge_ratio,
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
    path = tmp_path / f"gateway-m5-hedge-{port}.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def _start_gateway(config_path: Path, port: int) -> subprocess.Popen[str]:
    env = {
        **os.environ,
        "TIDEGATE_ADMIN_TOKEN": "dev-admin",
        "MOCK_A_KEY": "mock-key",
        "MOCK_B_KEY": "mock-key",
        "PYTHONPATH": f"{Path.cwd() / 'src'}:{Path.cwd()}",
    }
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
