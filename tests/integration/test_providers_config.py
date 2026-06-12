from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import yaml

from tests.integration.conftest import (
    API_KEY,
    BASE_URL,
    MOCK_A_URL,
    MOCK_B_URL,
    reset_mock,
    stats,
    wait_ready,
)


@pytest.mark.integration
def test_models_requires_bearer_and_lists_groups(gateway_proc: subprocess.Popen[str]) -> None:
    """SPEC-M1-3."""
    del gateway_proc
    with httpx.Client(timeout=2, trust_env=False) as client:
        unauthorized = client.get(f"{BASE_URL}/v1/models")
        response = client.get(
            f"{BASE_URL}/v1/models", headers={"Authorization": f"Bearer {API_KEY}"}
        )
    assert unauthorized.status_code == 401
    assert response.status_code == 200
    ids = {item["id"] for item in response.json()["data"]}
    assert "chat-large" in ids


@pytest.mark.integration
def test_ttft_timeout_switches_to_next_deployment(gateway_proc: subprocess.Popen[str]) -> None:
    """SPEC-M1-2."""
    del gateway_proc
    reset_mock(MOCK_A_URL)
    reset_mock(MOCK_B_URL)
    directive = {"ttft_ms": 800, "tpot_ms": 1, "output_tokens": 3}
    with httpx.Client(timeout=5, trust_env=False) as client:
        response = client.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "x-mock-directive": json.dumps(directive),
            },
            json={"model": "chat-large", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    assert stats(MOCK_A_URL)["aborted"] >= 0
    assert stats(MOCK_A_URL)["started"] + stats(MOCK_B_URL)["started"] >= 1


@pytest.mark.integration
def test_stream_drop_after_bytes_finishes_with_error_chunk(
    gateway_proc: subprocess.Popen[str],
) -> None:
    """SPEC-M1-1."""
    del gateway_proc
    directive = {
        "ttft_ms": 10,
        "tpot_ms": 1,
        "output_tokens": 10,
        "fail": "drop_after_n",
        "fail_n": 3,
    }
    chunks: list[str] = []
    with (
        httpx.Client(timeout=5, trust_env=False) as client,
        client.stream(
            "POST",
            f"{BASE_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "x-mock-directive": json.dumps(directive),
            },
            json={
                "model": "chat-large",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as response,
    ):
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("data:"):
                chunks.append(line)
    assert any('"finish_reason":"error"' in line for line in chunks)
    assert chunks[-1] == "data: [DONE]"


@pytest.mark.integration
def test_local_admin_reload_rolls_back_and_auth_cache_invalidates(
    mock_a_proc: subprocess.Popen[str],
    mock_b_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    """SPEC-M1-4."""
    del mock_a_proc, mock_b_proc
    config_path = tmp_path / "gateway.yaml"
    raw = yaml.safe_load(Path("tests/fixtures/gateway-test.yaml").read_text(encoding="utf-8"))
    raw["server"]["port"] = 8010
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    env = {
        **os.environ,
        "TIDEGATE_ADMIN_TOKEN": "dev-admin",
        "MOCK_A_KEY": "mock-key",
        "MOCK_B_KEY": "mock-key",
        "PYTHONPATH": f"{os.getcwd()}/src:{os.getcwd()}",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "tidegate", "--config", str(config_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        wait_ready("http://127.0.0.1:8010/healthz", proc)
        with httpx.Client(timeout=3, trust_env=False) as client:
            ok = client.get(
                "http://127.0.0.1:8010/v1/models",
                headers={"Authorization": f"Bearer {API_KEY}"},
            )
            assert ok.status_code == 200

            raw["tenants"] = []
            config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            reload_ok = client.post(
                "http://127.0.0.1:8010/admin/config/reload",
                headers={"X-Admin-Token": "dev-admin"},
            )
            assert reload_ok.status_code == 200
            denied = client.get(
                "http://127.0.0.1:8010/v1/models",
                headers={"Authorization": f"Bearer {API_KEY}"},
            )
            assert denied.status_code == 401

            raw["bad_extra_key"] = True
            config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            bad = client.post(
                "http://127.0.0.1:8010/admin/config/reload",
                headers={"X-Admin-Token": "dev-admin"},
            )
            assert bad.status_code == 422
            still_denied = client.get(
                "http://127.0.0.1:8010/v1/models",
                headers={"Authorization": f"Bearer {API_KEY}"},
            )
            assert still_denied.status_code == 401
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
