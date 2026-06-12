from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import redis

from tidegate.cache.embedding import embed_sync, init_embedding_worker
from tidegate.config.loader import load_config

BASE_URL = "http://127.0.0.1:8000"
MOCK_A_URL = "http://127.0.0.1:9001"
MOCK_B_URL = "http://127.0.0.1:9002"
API_KEY = "<demo key>"


def wait_ready(url: str, proc: subprocess.Popen[str] | None = None, timeout_s: float = 10) -> None:
    deadline = time.monotonic() + timeout_s
    with httpx.Client(timeout=0.5, trust_env=False) as client:
        while time.monotonic() < deadline:
            if proc is not None and proc.poll() is not None:
                _, stderr = proc.communicate()
                raise RuntimeError(f"service exited early: {stderr}")
            try:
                response = client.get(url)
                if response.status_code < 500:
                    return
            except httpx.HTTPError:
                time.sleep(0.1)
        raise RuntimeError(f"service did not become ready: {url}")


def _env() -> dict[str, str]:
    return {
        **os.environ,
        "TIDEGATE_ADMIN_TOKEN": "dev-admin",
        "MOCK_A_KEY": "mock-key",
        "MOCK_B_KEY": "mock-key",
        "PYTHONPATH": f"{os.getcwd()}/src:{os.getcwd()}",
    }


@pytest.fixture(scope="session")
def l2_model_ready() -> None:
    config = load_config("tests/fixtures/gateway-test.yaml")
    try:
        init_embedding_worker(
            config.cache.l2.embedding_model,
            config.cache.l2.model_cache_dir,
            config.cache.l2.hf_endpoint,
        )
        embed_sync(["模型预热"])
    except Exception as exc:
        pytest.skip(f"L2 fastembed model unavailable: {exc}")


@pytest.fixture(scope="session")
def redis_stack_proc() -> Iterator[None]:
    result = subprocess.run(
        ["docker", "compose", "-f", "deploy/docker-compose.yml", "up", "-d"],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        pytest.skip(f"docker redis unavailable: {result.stderr.strip()}")
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            if redis.Redis.from_url("redis://127.0.0.1:6379/0").ping():
                break
        except redis.RedisError:
            time.sleep(0.2)
    else:
        raise RuntimeError("redis-stack did not become ready")
    yield
    subprocess.run(["docker", "compose", "-f", "deploy/docker-compose.yml", "down"], check=True)


@pytest.fixture(autouse=True)
def clean_redis(redis_stack_proc: None) -> Iterator[None]:
    del redis_stack_proc
    client = redis.Redis.from_url("redis://127.0.0.1:6379/0")
    client.flushdb()
    yield
    redis.Redis.from_url("redis://127.0.0.1:6379/0").flushdb()


@pytest.fixture(scope="session")
def mock_a_proc() -> Iterator[subprocess.Popen[str]]:
    proc = subprocess.Popen(
        [sys.executable, "-m", "mock_provider", "--host", "127.0.0.1", "--port", "9001"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_env(),
    )
    wait_ready(f"{MOCK_A_URL}/__stats", proc)
    yield proc
    _stop(proc)


@pytest.fixture(scope="session")
def mock_b_proc() -> Iterator[subprocess.Popen[str]]:
    proc = subprocess.Popen(
        [sys.executable, "-m", "mock_provider", "--host", "127.0.0.1", "--port", "9002"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_env(),
    )
    wait_ready(f"{MOCK_B_URL}/__stats", proc)
    yield proc
    _stop(proc)


@pytest.fixture(scope="session")
def gateway_proc(
    redis_stack_proc: None,
    mock_a_proc: subprocess.Popen[str],
    mock_b_proc: subprocess.Popen[str],
) -> Iterator[subprocess.Popen[str]]:
    del redis_stack_proc, mock_a_proc, mock_b_proc
    proc = subprocess.Popen(
        [sys.executable, "-m", "tidegate", "--config", "tests/fixtures/gateway-test.yaml"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_env(),
    )
    wait_ready(f"{BASE_URL}/healthz", proc)
    yield proc
    _stop(proc)


@pytest.fixture
def editable_config(tmp_path: Path) -> Path:
    source = Path("tests/fixtures/gateway-test.yaml")
    target = tmp_path / "gateway.yaml"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def reset_mock(url: str) -> None:
    with httpx.Client(timeout=2, trust_env=False) as client:
        client.post(f"{url}/__reset")


def stats(url: str) -> dict[str, int]:
    with httpx.Client(timeout=2, trust_env=False) as client:
        payload = client.get(f"{url}/__stats").json()
    assert isinstance(payload, dict)
    return {str(key): int(value) for key, value in payload.items()}


def _stop(proc: subprocess.Popen[str]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
