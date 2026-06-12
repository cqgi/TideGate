from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import yaml

from tests.integration.conftest import API_KEY, MOCK_A_URL, MOCK_B_URL, reset_mock, wait_ready


@pytest.mark.integration
def test_failover_and_recovery(
    mock_a_proc: subprocess.Popen[str],
    mock_b_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    """SPEC-M3-2, SPEC-M3-3, and SPEC-M3-5."""
    del mock_a_proc, mock_b_proc
    reset_mock(MOCK_A_URL)
    reset_mock(MOCK_B_URL)
    _set_behavior(MOCK_A_URL, {"fail": "error_500"})
    port = 8030
    proc = _start_gateway(_m3_config(tmp_path, port), port)

    try:
        first_b_at: float | None = None
        routes: list[str] = []
        successes = 0
        started = time.monotonic()
        with httpx.Client(timeout=5, trust_env=False) as client:
            for _ in range(12):
                response = _chat(client, port, "chat-large")
                if response.status_code == 200:
                    successes += 1
                route = response.headers.get("X-TideGate-Route", "")
                routes.append(route)
                if route.startswith("mock-b/") and first_b_at is None:
                    first_b_at = time.monotonic()
                time.sleep(0.05)

            admin = client.get(
                f"http://127.0.0.1:{port}/admin/breakers",
                headers={"X-Admin-Token": "dev-admin"},
            )
            _set_behavior(MOCK_A_URL, {"fail": "none", "ttft_ms": 10})
            recovered = _wait_for_route(client, port, "mock-a/", timeout_s=3.0)
    finally:
        _stop_gateway(proc)

    assert first_b_at is not None
    assert first_b_at - started <= 2.0
    assert successes / len(routes) >= 0.95
    assert any(route.startswith("mock-b/") for route in routes)
    assert admin.status_code == 200
    assert admin.json()["breakers"]["mock-a/mock-gpt-large"]["state"] in {"open", "half_open"}
    assert recovered


@pytest.mark.integration
def test_slow_failure_opens_breaker(
    mock_a_proc: subprocess.Popen[str],
    mock_b_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    """SPEC-M3-1."""
    del mock_a_proc, mock_b_proc
    reset_mock(MOCK_A_URL)
    reset_mock(MOCK_B_URL)
    _set_behavior(MOCK_A_URL, {"ttft_ms": 250, "output_tokens": 1})
    port = 8031
    proc = _start_gateway(_m3_config(tmp_path, port, large_providers=("mock-a",)), port)

    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            for _ in range(8):
                _chat(client, port, "chat-large")
            admin = client.get(
                f"http://127.0.0.1:{port}/admin/breakers",
                headers={"X-Admin-Token": "dev-admin"},
            )
    finally:
        _stop_gateway(proc)

    assert admin.status_code == 200
    assert admin.json()["breakers"]["mock-a/mock-gpt-large"]["state"] in {"open", "half_open"}


@pytest.mark.integration
def test_rate_limit_cools_down_without_opening_breaker(
    mock_a_proc: subprocess.Popen[str],
    mock_b_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    """SPEC-M3-1."""
    del mock_a_proc, mock_b_proc
    reset_mock(MOCK_A_URL)
    reset_mock(MOCK_B_URL)
    _set_behavior(MOCK_A_URL, {"fail": "error_429", "retry_after_s": 2})
    port = 8032
    proc = _start_gateway(_m3_config(tmp_path, port), port)

    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            for _ in range(4):
                _chat(client, port, "chat-large")
            admin = client.get(
                f"http://127.0.0.1:{port}/admin/breakers",
                headers={"X-Admin-Token": "dev-admin"},
            )
            routes = [
                _chat(client, port, "chat-large").headers.get("X-TideGate-Route", "")
                for _ in range(3)
            ]
    finally:
        _stop_gateway(proc)

    breaker = admin.json()["breakers"]["mock-a/mock-gpt-large"]
    assert breaker["state"] == "closed"
    assert breaker["cooldown_until_s"] > 0
    assert all(route.startswith("mock-b/") for route in routes)


@pytest.mark.integration
def test_smaller_model_degradation_and_exhausted_502(
    mock_a_proc: subprocess.Popen[str],
    mock_b_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    """SPEC-M3-4."""
    del mock_a_proc, mock_b_proc
    reset_mock(MOCK_A_URL)
    reset_mock(MOCK_B_URL)
    _set_behavior(MOCK_A_URL, {"fail": "error_500"})
    port = 8033
    proc = _start_gateway(
        _m3_config(tmp_path, port, large_providers=("mock-a",), small_provider="mock-b"),
        port,
    )

    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            degraded = _chat(client, port, "chat-large")
            assert degraded.status_code == 200
            assert degraded.headers["X-TideGate-Degraded"] == "smaller-model"
            assert degraded.headers["X-TideGate-Route"] == "mock-b/mock-gpt-small"

            _set_behavior(MOCK_B_URL, {"fail": "error_500"})
            exhausted = _chat(client, port, "chat-large")
    finally:
        _stop_gateway(proc)

    assert exhausted.status_code == 502
    assert exhausted.json()["error"]["type"] == "upstream_error"


def _chat(client: httpx.Client, port: int, model: str) -> httpx.Response:
    return client.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
    )


def _set_behavior(url: str, behavior: dict[str, object]) -> None:
    with httpx.Client(timeout=2, trust_env=False) as client:
        response = client.post(f"{url}/__behavior", json=behavior)
    assert response.status_code == 200, response.text


def _wait_for_route(client: httpx.Client, port: int, prefix: str, *, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        response = _chat(client, port, "chat-large")
        if response.headers.get("X-TideGate-Route", "").startswith(prefix):
            return True
        time.sleep(0.1)
    return False


def _m3_config(
    tmp_path: Path,
    port: int,
    *,
    large_providers: tuple[str, ...] = ("mock-a", "mock-b"),
    small_provider: str = "mock-a",
) -> Path:
    raw = yaml.safe_load(Path("tests/fixtures/gateway-test.yaml").read_text(encoding="utf-8"))
    raw["server"]["port"] = port
    raw["quota_estimator"] = {"output_p95_fallback": 4, "correction_ewma_alpha": 0.1}
    raw["model_groups"]["chat-large"]["deployments"] = [
        deployment
        for deployment in raw["model_groups"]["chat-large"]["deployments"]
        if deployment["provider"] in large_providers
    ]
    raw["model_groups"]["chat-small"]["deployments"][0]["provider"] = small_provider
    raw["model_groups"]["chat-small"]["deployments"][0]["upstream_model"] = "mock-gpt-small"
    path = tmp_path / f"gateway-m3-{port}.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
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
