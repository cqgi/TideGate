from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import redis
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
def test_stream_exhausted_providers_is_in_band_error_and_non_stream_is_502(
    tmp_path: Path,
) -> None:
    """REWORK-M1-2."""
    config_path = tmp_path / "gateway.yaml"
    raw = yaml.safe_load(Path("tests/fixtures/gateway-test.yaml").read_text(encoding="utf-8"))
    raw["server"]["port"] = 8011
    raw["timeouts"] = {**raw["timeouts"], "connect_s": 0.1, "ttft_s": 0.1, "total_s": 2.0}
    raw["providers"]["mock-a"]["base_url"] = "http://127.0.0.1:9/v1"
    raw["providers"]["mock-b"]["base_url"] = "http://127.0.0.1:9/v1"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    proc = _start_gateway(config_path, 8011)
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            with client.stream(
                "POST",
                "http://127.0.0.1:8011/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": "chat-large",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            ) as response:
                chunks = [line for line in response.iter_lines() if line.startswith("data:")]
                assert response.status_code == 200

            non_stream = client.post(
                "http://127.0.0.1:8011/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={"model": "chat-large", "messages": [{"role": "user", "content": "hi"}]},
            )
    finally:
        _stop_gateway(proc)

    assert any('"finish_reason":"error"' in line for line in chunks)
    assert chunks[-1] == "data: [DONE]"
    assert non_stream.status_code == 502


@pytest.mark.integration
def test_non_stream_allows_body_slower_than_inter_chunk(
    mock_a_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    """REWORK-M1-3."""
    del mock_a_proc
    config_path = tmp_path / "gateway.yaml"
    raw = yaml.safe_load(Path("tests/fixtures/gateway-test.yaml").read_text(encoding="utf-8"))
    raw["server"]["port"] = 8012
    raw["timeouts"] = {**raw["timeouts"], "inter_chunk_s": 0.01, "total_s": 3.0}
    raw["model_groups"]["chat-large"]["deployments"] = [
        raw["model_groups"]["chat-large"]["deployments"][0]
    ]
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    proc = _start_gateway(config_path, 8012)
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            response = client.post(
                "http://127.0.0.1:8012/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "x-mock-directive": json.dumps(
                        {"ttft_ms": 80, "tpot_ms": 1, "output_tokens": 1}
                    ),
                },
                json={"model": "chat-large", "messages": [{"role": "user", "content": "hi"}]},
            )
    finally:
        _stop_gateway(proc)

    assert response.status_code == 200
    assert response.json()["usage"]["completion_tokens"] == 1


@pytest.mark.integration
def test_config_poll_recovers_after_redis_restart(
    redis_stack_proc: None,
    mock_a_proc: subprocess.Popen[str],
    mock_b_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    """REWORK-M1-5."""
    del redis_stack_proc, mock_a_proc, mock_b_proc
    redis_client = redis.Redis.from_url("redis://127.0.0.1:6379/0")
    redis_client.delete("cfg:version")
    config_path = tmp_path / "gateway.yaml"
    raw = yaml.safe_load(Path("tests/fixtures/gateway-test.yaml").read_text(encoding="utf-8"))
    raw["server"]["port"] = 8013
    raw["server"]["config_poll_interval_s"] = 0.2
    raw["server"]["config_reload_backoff_initial_s"] = 0.1
    raw["server"]["config_reload_backoff_max_s"] = 0.2
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    proc = _start_gateway(config_path, 8013)
    try:
        with httpx.Client(timeout=3, trust_env=False) as client:
            before = client.get(
                "http://127.0.0.1:8013/v1/models",
                headers={"Authorization": f"Bearer {API_KEY}"},
            )
            assert before.status_code == 200

        subprocess.run(
            ["docker", "compose", "-f", "deploy/docker-compose.yml", "restart", "redis-stack"],
            check=True,
            text=True,
            capture_output=True,
        )
        wait_ready("http://127.0.0.1:9001/__stats")
        _wait_for_redis(redis_client)

        raw["tenants"] = []
        config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        redis_client.set("cfg:version", "1")

        deadline = time.monotonic() + 10
        with httpx.Client(timeout=3, trust_env=False) as client:
            while time.monotonic() < deadline:
                denied = client.get(
                    "http://127.0.0.1:8013/v1/models",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                )
                if denied.status_code == 401:
                    return
                time.sleep(0.2)
        raise AssertionError("polling reload did not recover after redis restart")
    finally:
        _stop_gateway(proc)


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


def _start_gateway(config_path: Path, port: int) -> subprocess.Popen[str]:
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
    wait_ready(f"http://127.0.0.1:{port}/healthz", proc)
    return proc


def _stop_gateway(proc: subprocess.Popen[str]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _wait_for_redis(redis_client: redis.Redis) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            if redis_client.ping():
                return
        except redis.RedisError:
            time.sleep(0.2)
    raise AssertionError("redis did not recover after restart")
