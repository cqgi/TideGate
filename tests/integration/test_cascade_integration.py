from __future__ import annotations

import os
import subprocess
import sys
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
def test_cascade_accepts_high_confidence_draft(
    mock_a_proc: subprocess.Popen[str],
    mock_b_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    """SPEC-M5-2."""
    del mock_a_proc, mock_b_proc
    reset_mock(MOCK_A_URL)
    reset_mock(MOCK_B_URL)
    _set_behavior(MOCK_A_URL, {"logprob_mean": -0.2, "output_tokens": 2})
    port = 8050
    proc = _start_gateway(_cascade_config(tmp_path, port), port)

    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            response = _chat(client, port)
            mock_a_started = stats(MOCK_A_URL)["started"]
            mock_b_started = stats(MOCK_B_URL)["started"]
            metrics = client.get(f"http://127.0.0.1:{port}/metrics").text
    finally:
        _stop_gateway(proc)

    assert response.status_code == 200, response.text
    assert response.headers["X-TideGate-Route"] == "mock-a/mock-gpt-small"
    assert mock_a_started == 1
    assert mock_b_started == 0
    assert 'tidegate_cascade_total{outcome="draft_accepted"} 1.0' in metrics


@pytest.mark.integration
def test_cascade_escalates_low_confidence_draft(
    mock_a_proc: subprocess.Popen[str],
    mock_b_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    """SPEC-M5-2."""
    del mock_a_proc, mock_b_proc
    reset_mock(MOCK_A_URL)
    reset_mock(MOCK_B_URL)
    _set_behavior(MOCK_A_URL, {"logprob_mean": -0.9, "output_tokens": 2})
    port = 8051
    proc = _start_gateway(_cascade_config(tmp_path, port), port)

    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            response = _chat(client, port)
            mock_a_started = stats(MOCK_A_URL)["started"]
            mock_b_started = stats(MOCK_B_URL)["started"]
            metrics = client.get(f"http://127.0.0.1:{port}/metrics").text
    finally:
        _stop_gateway(proc)

    assert response.status_code == 200, response.text
    assert response.headers["X-TideGate-Route"] == "mock-b/mock-gpt-large"
    assert mock_a_started == 1
    assert mock_b_started == 1
    assert 'tidegate_cascade_total{outcome="escalated"} 1.0' in metrics


@pytest.mark.integration
def test_cascade_bypasses_stream_requests(
    mock_a_proc: subprocess.Popen[str],
    mock_b_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    """SPEC-M5-2."""
    del mock_a_proc, mock_b_proc
    reset_mock(MOCK_A_URL)
    reset_mock(MOCK_B_URL)
    _set_behavior(MOCK_A_URL, {"logprob_mean": -0.2, "output_tokens": 2})
    port = 8052
    proc = _start_gateway(_cascade_config(tmp_path, port), port)

    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            response = _chat(client, port, stream=True)
            content = response.text
            mock_a_started = stats(MOCK_A_URL)["started"]
            mock_b_started = stats(MOCK_B_URL)["started"]
            metrics = client.get(f"http://127.0.0.1:{port}/metrics").text
    finally:
        _stop_gateway(proc)

    assert response.status_code == 200, content
    assert response.headers["X-TideGate-Route"] == "mock-b/mock-gpt-large"
    assert mock_a_started == 0
    assert mock_b_started == 1
    assert 'tidegate_cascade_total{outcome="bypassed"} 1.0' in metrics


def _chat(client: httpx.Client, port: int, *, stream: bool = False) -> httpx.Response:
    return client.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": "chat-large",
            "stream": stream,
            "messages": [{"role": "user", "content": f"cascade test {port}"}],
        },
    )


def _set_behavior(url: str, behavior: dict[str, object]) -> None:
    with httpx.Client(timeout=2, trust_env=False) as client:
        response = client.post(f"{url}/__behavior", json=behavior)
    assert response.status_code == 200, response.text


def _cascade_config(tmp_path: Path, port: int) -> Path:
    raw = yaml.safe_load(Path("tests/fixtures/gateway-test.yaml").read_text(encoding="utf-8"))
    raw["server"]["port"] = port
    raw["policies"]["default"]["cascade"] = {
        "enabled": True,
        "draft_model_group": "chat-small",
        "confidence_metric": "mean_logprob",
        "threshold": -0.45,
    }
    raw["model_groups"]["chat-small"]["deployments"][0]["provider"] = "mock-a"
    raw["model_groups"]["chat-small"]["deployments"][0]["upstream_model"] = "mock-gpt-small"
    raw["model_groups"]["chat-large"]["deployments"] = [
        {
            "provider": "mock-b",
            "upstream_model": "mock-gpt-large",
            "weight": 1,
            "price_per_1k_input_usd": 0.0025,
            "price_per_1k_output_usd": 0.005,
            "supports_logprobs": True,
        }
    ]
    raw["tenants"][0]["cache"] = {"l1": False, "l2": False}
    path = tmp_path / f"gateway-m5-cascade-{port}.yaml"
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
