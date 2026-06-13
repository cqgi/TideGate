from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import httpx
import pytest
import yaml
from openai import OpenAI

from tests.integration.conftest import (
    API_KEY,
    BASE_URL,
    MOCK_A_URL,
    MOCK_B_URL,
    reset_mock,
    stats,
    wait_ready,
)
from tidegate.quota.keys import rpm_key, tpm_key


@pytest.mark.integration
def test_l1_second_request_hit_exact_and_quota_refund(
    gateway_proc: subprocess.Popen[str],
) -> None:
    del gateway_proc
    reset_mock(MOCK_A_URL)
    with httpx.Client(timeout=2, trust_env=False) as client:
        client.post(f"{MOCK_A_URL}/__behavior", json={"ttft_ms": 1, "output_tokens": 1})
    body: dict[str, object] = {
        "model": "chat-large",
        "messages": [{"role": "user", "content": "怎么退款"}],
    }
    with httpx.Client(timeout=5, trust_env=False) as client:
        first = _chat(client, body)
        before_stats = stats(MOCK_A_URL)["started"]
        before_quota = _quota()
        started = time.monotonic()
        second = _chat(client, body)
        elapsed = time.monotonic() - started
        after_quota = _quota()
        after_stats = stats(MOCK_A_URL)["started"]

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.headers["X-TideGate-Cache"] == "hit-exact"
    assert second.headers["X-TideGate-Route"] == "cache"
    assert elapsed < 0.05
    assert after_stats == before_stats
    rpm_delta = before_quota["rpm"] - after_quota["rpm"]
    assert 0.65 <= rpm_delta <= 1.05
    assert after_quota["tpm"] >= before_quota["tpm"] - 1.0


@pytest.mark.integration
def test_l1_temperature_changes_key(gateway_proc: subprocess.Popen[str]) -> None:
    del gateway_proc
    reset_mock(MOCK_A_URL)
    with httpx.Client(timeout=5, trust_env=False) as client:
        first = _chat(client, {"model": "chat-large", "temperature": 0.1, "messages": [_msg("hi")]})
        second = _chat(
            client, {"model": "chat-large", "temperature": 0.2, "messages": [_msg("hi")]}
        )

    assert first.headers["X-TideGate-Cache"] == "miss"
    assert second.headers["X-TideGate-Cache"] == "miss"


@pytest.mark.integration
def test_singleflight_collapses_concurrent_miss(gateway_proc: subprocess.Popen[str]) -> None:
    del gateway_proc
    reset_mock(MOCK_A_URL)
    reset_mock(MOCK_B_URL)
    with httpx.Client(timeout=2, trust_env=False) as client:
        client.post(f"{MOCK_A_URL}/__behavior", json={"ttft_ms": 200, "output_tokens": 1})
        client.post(f"{MOCK_B_URL}/__behavior", json={"ttft_ms": 200, "output_tokens": 1})

    body: dict[str, object] = {
        "model": "chat-large",
        "messages": [{"role": "user", "content": "并发退款"}],
    }
    before = _started_total()
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as pool:
        responses = list(pool.map(lambda _: _chat_once(body), range(50)))
    after = _started_total()

    assert all(response.status_code == 200 for response in responses)
    assert after - before == 1
    assert (
        sum(1 for response in responses if response.headers["X-TideGate-Cache"] == "hit-exact")
        >= 49
    )


@pytest.mark.integration
def test_stream_replay_from_cache_openai_sdk(gateway_proc: subprocess.Popen[str]) -> None:
    del gateway_proc
    reset_mock(MOCK_A_URL)
    headers = {"x-mock-directive": json.dumps({"ttft_ms": 10, "output_tokens": 6})}
    client = OpenAI(
        api_key=API_KEY,
        base_url=f"{BASE_URL}/v1",
        default_headers=headers,
        http_client=httpx.Client(trust_env=False),
    )
    first = client.chat.completions.create(
        model="chat-large",
        messages=[{"role": "user", "content": "流式缓存"}],
    )
    stream = client.chat.completions.create(
        model="chat-large",
        stream=True,
        stream_options={"include_usage": True},
        messages=[{"role": "user", "content": "流式缓存"}],
    )
    content = ""
    usage_seen = False
    for chunk in stream:
        if chunk.usage is not None:
            usage_seen = True
        if chunk.choices:
            content += chunk.choices[0].delta.content or ""

    assert content == first.choices[0].message.content
    assert usage_seen


@pytest.mark.integration
def test_stream_success_populates_l1_cache(gateway_proc: subprocess.Popen[str]) -> None:
    del gateway_proc
    reset_mock(MOCK_A_URL)
    headers = {"x-mock-directive": json.dumps({"ttft_ms": 10, "output_tokens": 4})}
    client = OpenAI(
        api_key=API_KEY,
        base_url=f"{BASE_URL}/v1",
        default_headers=headers,
        http_client=httpx.Client(trust_env=False),
    )
    stream = client.chat.completions.create(
        model="chat-large",
        stream=True,
        stream_options={"include_usage": True},
        messages=[{"role": "user", "content": "流式首次写缓存"}],
    )
    first_content = ""
    for chunk in stream:
        if chunk.choices:
            first_content += chunk.choices[0].delta.content or ""

    with httpx.Client(timeout=5, trust_env=False) as raw_client:
        before = _started_total()
        second = _chat(
            raw_client,
            {
                "model": "chat-large",
                "messages": [{"role": "user", "content": "流式首次写缓存"}],
            },
        )
        after = _started_total()

    assert first_content
    assert second.status_code == 200
    assert second.headers["X-TideGate-Cache"] == "hit-exact"
    assert after == before


@pytest.mark.integration
def test_stream_singleflight_collapses_concurrent_miss(
    gateway_proc: subprocess.Popen[str],
) -> None:
    del gateway_proc
    reset_mock(MOCK_A_URL)
    reset_mock(MOCK_B_URL)
    with httpx.Client(timeout=2, trust_env=False) as client:
        client.post(f"{MOCK_A_URL}/__behavior", json={"ttft_ms": 200, "output_tokens": 3})
        client.post(f"{MOCK_B_URL}/__behavior", json={"ttft_ms": 200, "output_tokens": 3})

    before = _started_total()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        contents = list(pool.map(lambda _: _stream_content_once("并发流式缓存"), range(5)))
    after = _started_total()

    assert all(content == contents[0] for content in contents)
    assert contents[0]
    assert after - before == 1


@pytest.mark.integration
def test_volatile_intent_bypasses_cache(gateway_proc: subprocess.Popen[str]) -> None:
    del gateway_proc
    reset_mock(MOCK_A_URL)
    body: dict[str, object] = {
        "model": "chat-large",
        "messages": [{"role": "user", "content": "今天北京天气怎么样"}],
    }
    with httpx.Client(timeout=5, trust_env=False) as client:
        first = _chat(client, body)
        before = _started_total()
        second = _chat(client, body)
        after = _started_total()

    assert first.headers["X-TideGate-Cache"] == "bypass"
    assert second.headers["X-TideGate-Cache"] == "bypass"
    assert after == before + 1


@pytest.mark.integration
def test_disabled_l1_bypasses_cache_and_singleflight(
    mock_a_proc: subprocess.Popen[str],
    mock_b_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    del mock_a_proc, mock_b_proc
    reset_mock(MOCK_A_URL)
    reset_mock(MOCK_B_URL)
    port = 8045
    proc = _start_gateway(_m4_l1_disabled_config(tmp_path, port), port)

    try:
        body: dict[str, object] = {
            "model": "chat-large",
            "messages": [{"role": "user", "content": "禁用缓存并发"}],
        }
        before = _started_total()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            responses = list(
                pool.map(
                    lambda _: _chat_port_once(port, body),
                    range(5),
                )
            )
        after = _started_total()
    finally:
        _stop_gateway(proc)

    assert all(response.status_code == 200 for response in responses)
    assert all(response.headers["X-TideGate-Cache"] == "bypass" for response in responses)
    assert after == before + 5


def _chat_once(body: Mapping[str, object]) -> httpx.Response:
    with httpx.Client(timeout=5, trust_env=False) as client:
        return _chat(client, body)


def _chat(client: httpx.Client, body: Mapping[str, object]) -> httpx.Response:
    return client.post(
        f"{BASE_URL}/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=body,
    )


def _msg(content: str) -> dict[str, str]:
    return {"role": "user", "content": content}


def _quota() -> dict[str, float]:
    import redis

    client = redis.Redis.from_url("redis://127.0.0.1:6379/0")
    return {
        "rpm": float(client.hget(rpm_key("demo"), "tokens") or 0),
        "tpm": float(client.hget(tpm_key("demo"), "tokens") or 0),
    }


def _started_total() -> int:
    return stats(MOCK_A_URL)["started"] + stats(MOCK_B_URL)["started"]


def _stream_content_once(content: str) -> str:
    client = OpenAI(
        api_key=API_KEY,
        base_url=f"{BASE_URL}/v1",
        http_client=httpx.Client(timeout=5, trust_env=False),
    )
    stream = client.chat.completions.create(
        model="chat-large",
        stream=True,
        stream_options={"include_usage": True},
        messages=[{"role": "user", "content": content}],
    )
    result = ""
    for chunk in stream:
        if chunk.choices:
            result += chunk.choices[0].delta.content or ""
    return result


def _chat_port_once(port: int, body: Mapping[str, object]) -> httpx.Response:
    with httpx.Client(timeout=5, trust_env=False) as client:
        return client.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=body,
        )


def _m4_l1_disabled_config(tmp_path: Path, port: int) -> Path:
    raw = yaml.safe_load(Path("tests/fixtures/gateway-test.yaml").read_text(encoding="utf-8"))
    raw["server"]["port"] = port
    raw["tenants"][0]["cache"] = {"l1": False, "l2": False}
    path = tmp_path / f"gateway-m4-l1-disabled-{port}.yaml"
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
