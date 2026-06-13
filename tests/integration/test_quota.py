from __future__ import annotations

import asyncio
import json
import os
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import redis
import yaml

from tests.integration.conftest import API_KEY, MOCK_A_URL, reset_mock, wait_ready
from tidegate.config.loader import load_config
from tidegate.config.models import DeploymentConfig, GatewayConfig
from tidegate.core.models import ChatMessage, UnifiedRequest, Usage
from tidegate.quota.estimator import Estimate, QuotaEstimator, RedisCorrectionStore
from tidegate.quota.keys import conc_key, reservation_zset_key, rpm_key
from tidegate.quota.scripts import QuotaScripts
from tidegate.quota.service import QuotaService


@pytest.mark.integration
@pytest.mark.parametrize(
    ("dim", "plan_patch", "body_patch", "expected_code"),
    [
        ("rpm", {"rpm": 0}, {}, "rpm_exceeded"),
        ("tpm", {"tpm": 1}, {"max_tokens": 2000}, "tpm_exceeded"),
        ("concurrency", {"concurrent_streams": 0}, {}, "concurrency_exceeded"),
        ("budget", {"monthly_budget_usd": 0}, {}, "budget_exceeded"),
    ],
)
def test_quota_rejects_each_dimension(
    dim: str,
    plan_patch: dict[str, int | float | str],
    body_patch: dict[str, int],
    expected_code: str,
    mock_a_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    del mock_a_proc
    ports = {"rpm": 8020, "tpm": 8021, "concurrency": 8022, "budget": 8023}
    port = ports[dim]
    config_path = _quota_config(tmp_path, port=port, plan_patch=plan_patch, single_provider=True)
    proc = _start_gateway(config_path, port)
    try:
        body = {
            "model": "chat-large",
            "messages": [{"role": "user", "content": "hi"}],
        } | body_patch
        with httpx.Client(timeout=5, trust_env=False) as client:
            response = client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=body,
            )
    finally:
        _stop_gateway(proc)

    assert response.status_code == 429
    assert response.headers["Retry-After"].isdigit()
    assert response.json()["error"]["code"] == expected_code


@pytest.mark.integration
def test_quota_settlement_refunds_estimate_delta(
    mock_a_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    del mock_a_proc
    reset_mock(MOCK_A_URL)
    port = 8024
    config_path = _quota_config(tmp_path, port=port, single_provider=True)
    proc = _start_gateway(config_path, port)
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            response = client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "x-mock-directive": json.dumps(
                        {"ttft_ms": 10, "tpot_ms": 1, "output_tokens": 50}
                    ),
                },
                json={
                    "model": "chat-large",
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            debug = client.get(
                f"http://127.0.0.1:{port}/metrics",
                headers={"Authorization": f"Bearer {API_KEY}"},
            )
    finally:
        _stop_gateway(proc)

    assert response.status_code == 200
    assert response.json()["usage"]["completion_tokens"] == 50
    assert 'tidegate_tokens_total{direction="out",model="chat-large",tenant="demo"} 50.0' in (
        debug.text
    )


@pytest.mark.integration
def test_stream_disconnect_releases_concurrency(
    mock_a_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    del mock_a_proc
    reset_mock(MOCK_A_URL)
    port = 8027
    config_path = _quota_config(tmp_path, port=port, single_provider=True)
    proc = _start_gateway(config_path, port)
    body = json.dumps(
        {
            "model": "chat-large",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    directive = {"ttft_ms": 10, "tpot_ms": 80, "output_tokens": 50}
    request_bytes = (
        "POST /v1/chat/completions HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{port}\r\n"
        f"Authorization: Bearer {API_KEY}\r\n"
        f"x-mock-directive: {json.dumps(directive)}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body.encode())}\r\n"
        "Connection: close\r\n"
        "\r\n"
        f"{body}"
    ).encode()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
            sock.sendall(request_bytes)
            sock.settimeout(5)
            data = b""
            while data.count(b"data:") < 3:
                data += sock.recv(1024)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            sock.shutdown(socket.SHUT_RDWR)
        _wait_for_quota_release()
    finally:
        _stop_gateway(proc)


@pytest.mark.integration
def test_stream_retry_charges_single_rpm(
    mock_a_proc: subprocess.Popen[str],
    mock_b_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    del mock_a_proc, mock_b_proc
    reset_mock(MOCK_A_URL)
    port = 8028
    config_path = _quota_config(
        tmp_path,
        port=port,
        single_provider=False,
        extra_updates={
            "timeouts": {"connect_s": 0.1, "ttft_s": 1.0, "inter_chunk_s": 15.0, "total_s": 5.0},
            "routing": {"max_attempts_before_first_byte": 2},
            "providers": {
                "mock-a": {
                    "type": "openai_compatible",
                    "base_url": "http://127.0.0.1:9/v1",
                    "api_key_env": "MOCK_A_KEY",
                    "max_connections": 200,
                }
            },
        },
    )
    proc = _start_gateway(config_path, port)
    try:
        with (
            httpx.Client(timeout=5, trust_env=False) as client,
            client.stream(
                "POST",
                f"http://127.0.0.1:{port}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                },
                json={
                    "model": "chat-large",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            ) as response,
        ):
            lines = [line for line in response.iter_lines() if line.startswith("data:")]
            assert response.status_code == 200
        state = _rpm_and_reservations()
    finally:
        _stop_gateway(proc)

    assert lines[-1] == "data: [DONE]"
    assert state["rpm_tokens"] == pytest.approx(599.0, abs=0.2)
    assert state["resv"] == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_service_double_instance_does_not_oversell(redis_stack_proc: None) -> None:
    del redis_stack_proc
    snapshot = load_config(Path("tests/fixtures/gateway-test.yaml")).model_copy(
        update={
            "quota_plans": {
                "free": load_config(Path("tests/fixtures/gateway-test.yaml"))
                .quota_plans["free"]
                .model_copy(update={"rpm": 1000, "tpm": 10, "concurrent_streams": 100})
            }
        }
    )
    client_a = redis.asyncio.Redis.from_url("redis://127.0.0.1:6379/0", decode_responses=False)
    client_b = redis.asyncio.Redis.from_url("redis://127.0.0.1:6379/0", decode_responses=False)
    services = [
        await _service(
            client_a, Estimate(prompt_tokens=1, output_tokens=3, tpm_cost=3, budget_cost_micro=1)
        ),
        await _service(
            client_b, Estimate(prompt_tokens=1, output_tokens=3, tpm_cost=3, budget_cost_micro=1)
        ),
    ]
    tenant = snapshot.tenants[0]
    deployment = snapshot.model_groups["chat-large"].deployments[0]

    async def reserve(index: int) -> bool:
        req = _req(f"req-{index}", max_tokens=3)
        try:
            await services[index % 2].reserve(
                tenant=tenant,
                req=req,
                deployment=deployment,
                snapshot=snapshot,
            )
        except Exception:
            return False
        return True

    results = await asyncio.gather(*(reserve(index) for index in range(10)))
    await client_a.aclose()
    await client_b.aclose()
    assert sum(results) <= 4


@pytest.mark.integration
def test_redis_down_fallback_open_and_closed(
    redis_stack_proc: None,
    mock_a_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    del redis_stack_proc, mock_a_proc
    open_path = _quota_config(
        tmp_path,
        port=8025,
        plan_patch={"fail_mode": "open"},
        single_provider=True,
    )
    closed_path = _quota_config(
        tmp_path, port=8026, plan_patch={"fail_mode": "closed"}, single_provider=True
    )
    open_proc = _start_gateway(open_path, 8025)
    closed_proc = _start_gateway(closed_path, 8026)
    subprocess.run(
        ["docker", "compose", "-f", "deploy/docker-compose.yml", "stop", "redis-stack"],
        check=True,
        text=True,
        capture_output=True,
    )
    try:
        with httpx.Client(timeout=5, trust_env=False) as client:
            open_response = client.post(
                "http://127.0.0.1:8025/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={"model": "chat-large", "messages": [{"role": "user", "content": "hi"}]},
            )
            closed_response = client.post(
                "http://127.0.0.1:8026/v1/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={"model": "chat-large", "messages": [{"role": "user", "content": "hi"}]},
            )
    finally:
        _stop_gateway(open_proc)
        _stop_gateway(closed_proc)
        subprocess.run(
            ["docker", "compose", "-f", "deploy/docker-compose.yml", "start", "redis-stack"],
            check=True,
            text=True,
            capture_output=True,
        )
        _wait_for_redis()

    assert open_response.status_code == 200
    assert closed_response.status_code == 503


async def _service(client: redis.asyncio.Redis, estimate: Estimate | None = None) -> QuotaService:
    scripts = QuotaScripts(client)
    await scripts.load()
    estimator = (
        FixedEstimator(estimate)
        if estimate is not None
        else QuotaEstimator(RedisCorrectionStore(client))
    )
    return QuotaService(client, scripts, estimator)


class FixedEstimator(QuotaEstimator):
    def __init__(self, estimate: Estimate) -> None:
        self._estimate = estimate

    async def estimate(
        self,
        req: UnifiedRequest,
        deployment: DeploymentConfig,
        snapshot: GatewayConfig,
    ) -> Estimate:
        del req, deployment, snapshot
        return self._estimate

    async def update_correction(
        self,
        *,
        tenant_id: str,
        model: str,
        estimate: Estimate,
        actual: Usage,
        snapshot: GatewayConfig,
    ) -> None:
        del tenant_id, model, estimate, actual, snapshot


def _req(request_id: str, max_tokens: int | None = 3) -> UnifiedRequest:
    return UnifiedRequest(
        request_id=request_id,
        tenant_id="demo",
        model="chat-large",
        messages=[ChatMessage(role="user", content="hi")],
        stream=False,
        max_tokens=max_tokens,
        raw_body={"model": "chat-large", "messages": [{"role": "user", "content": "hi"}]},
    )


def _quota_config(
    tmp_path: Path,
    *,
    port: int,
    plan_patch: dict[str, int | float | str] | None = None,
    single_provider: bool = False,
    extra_updates: dict[str, object] | None = None,
) -> Path:
    raw = yaml.safe_load(Path("tests/fixtures/gateway-test.yaml").read_text(encoding="utf-8"))
    raw["server"]["port"] = port
    raw.setdefault("quota_estimator", {})["output_p95_fallback"] = 8
    raw.setdefault("sweeper", {})["interval_s"] = 1
    raw["quota_plans"]["free"] = raw["quota_plans"]["free"] | (plan_patch or {})
    if extra_updates:
        for section, patch in extra_updates.items():
            if isinstance(patch, dict) and isinstance(raw.get(section), dict):
                raw[section] = _deep_merge(raw[section], patch)
            else:
                raw[section] = patch
    if single_provider:
        raw["model_groups"]["chat-large"]["deployments"] = [
            raw["model_groups"]["chat-large"]["deployments"][0]
        ]
    path = tmp_path / f"gateway-{port}.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _deep_merge(base: dict[str, object], patch: dict[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in patch.items():
        current = merged.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


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


def _wait_for_redis() -> None:
    client = redis.Redis.from_url("redis://127.0.0.1:6379/0")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            if client.ping():
                return
        except redis.RedisError:
            time.sleep(0.2)
    raise AssertionError("redis did not restart")


def _wait_for_quota_release() -> None:
    client = redis.Redis.from_url("redis://127.0.0.1:6379/0")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        conc = int(client.get(conc_key("demo")) or 0)
        resv = int(client.zcard(reservation_zset_key("demo")))
        if conc == 0 and resv == 0:
            return
        time.sleep(0.1)
    raise AssertionError("quota reservation was not released after disconnect")


def _rpm_and_reservations() -> dict[str, float | int]:
    client = redis.Redis.from_url("redis://127.0.0.1:6379/0")
    raw_tokens = client.hget(rpm_key("demo"), "tokens")
    assert raw_tokens is not None
    return {
        "rpm_tokens": float(raw_tokens),
        "resv": int(client.zcard(reservation_zset_key("demo"))),
    }
