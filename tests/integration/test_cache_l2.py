from __future__ import annotations

import hashlib
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

OTHER_API_KEY = "<other key>"
AGGRESSIVE_API_KEY = "<aggressive key>"
CONSERVATIVE_API_KEY = "<conservative key>"


@pytest.mark.integration
def test_l2_paraphrase_hits_semantic(
    l2_model_ready: None,
    mock_a_proc: subprocess.Popen[str],
    mock_b_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    del l2_model_ready, mock_a_proc, mock_b_proc
    reset_mock(MOCK_A_URL)
    reset_mock(MOCK_B_URL)
    port = 8040
    proc = _start_gateway(_m4_l2_config(tmp_path, port), port)

    try:
        with httpx.Client(timeout=10, trust_env=False) as client:
            first = _chat(client, port, "怎么申请退款")
            before = stats(MOCK_A_URL)["started"] + stats(MOCK_B_URL)["started"]
            second = _chat(client, port, "退款流程是什么")
            after = stats(MOCK_A_URL)["started"] + stats(MOCK_B_URL)["started"]
    finally:
        _stop_gateway(proc)

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert second.headers["X-TideGate-Cache"] == "hit-semantic"
    assert second.headers["X-TideGate-Route"] == "cache"
    assert after == before


@pytest.mark.integration
def test_l2_multiturn_is_bypassed(
    l2_model_ready: None,
    mock_a_proc: subprocess.Popen[str],
    mock_b_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    del l2_model_ready, mock_a_proc, mock_b_proc
    reset_mock(MOCK_A_URL)
    reset_mock(MOCK_B_URL)
    port = 8041
    proc = _start_gateway(_m4_l2_config(tmp_path, port), port)

    try:
        with httpx.Client(timeout=10, trust_env=False) as client:
            first = _chat(client, port, "怎么申请退款")
            before = stats(MOCK_A_URL)["started"] + stats(MOCK_B_URL)["started"]
            second = _chat_messages(
                client,
                port,
                [
                    {"role": "user", "content": "怎么申请退款"},
                    {"role": "assistant", "content": "请说明订单情况"},
                    {"role": "user", "content": "退款流程是什么"},
                ],
            )
            after = stats(MOCK_A_URL)["started"] + stats(MOCK_B_URL)["started"]
    finally:
        _stop_gateway(proc)

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert second.headers["X-TideGate-Cache"] == "miss"
    assert after == before + 1


@pytest.mark.integration
def test_l2_isolates_tenant_and_prompt_version(
    l2_model_ready: None,
    mock_a_proc: subprocess.Popen[str],
    mock_b_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    del l2_model_ready, mock_a_proc, mock_b_proc
    reset_mock(MOCK_A_URL)
    reset_mock(MOCK_B_URL)
    port = 8042
    proc = _start_gateway(_m4_l2_config(tmp_path, port, add_other_tenant=True), port)

    try:
        with httpx.Client(timeout=10, trust_env=False) as client:
            seed = _chat(client, port, "怎么申请退款")
            before_other = _started_total()
            other_tenant = _chat(client, port, "退款流程是什么", api_key=OTHER_API_KEY)
            after_other = _started_total()
            before_prompt = _started_total()
            other_prompt = _chat(
                client,
                port,
                "退款流程是什么",
                prompt_version="v2",
            )
            after_prompt = _started_total()
    finally:
        _stop_gateway(proc)

    assert seed.status_code == 200, seed.text
    assert other_tenant.status_code == 200, other_tenant.text
    assert other_tenant.headers["X-TideGate-Cache"] == "miss"
    assert after_other == before_other + 1
    assert other_prompt.status_code == 200, other_prompt.text
    assert other_prompt.headers["X-TideGate-Cache"] == "miss"
    assert after_prompt == before_prompt + 1


@pytest.mark.integration
def test_feedback_evicts_semantic_entry(
    l2_model_ready: None,
    mock_a_proc: subprocess.Popen[str],
    mock_b_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    del l2_model_ready, mock_a_proc, mock_b_proc
    reset_mock(MOCK_A_URL)
    reset_mock(MOCK_B_URL)
    port = 8043
    proc = _start_gateway(_m4_l2_config(tmp_path, port), port)

    try:
        with httpx.Client(timeout=10, trust_env=False) as client:
            seed = _chat(client, port, "怎么申请退款")
            semantic = _chat(client, port, "退款流程是什么")
            feedback = client.post(
                f"http://127.0.0.1:{port}/v1/cache/feedback",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={"request_id": semantic.headers["X-Request-Id"], "verdict": "wrong_answer"},
            )
            before = _started_total()
            after_feedback = _chat(client, port, "退款流程是什么")
            after = _started_total()
    finally:
        _stop_gateway(proc)

    assert seed.status_code == 200, seed.text
    assert semantic.headers["X-TideGate-Cache"] == "hit-semantic"
    assert feedback.status_code == 200, feedback.text
    assert feedback.json()["evicted"] is True
    assert after_feedback.headers["X-TideGate-Cache"] == "miss"
    assert after == before + 1


@pytest.mark.integration
def test_stale_cache_degrades_after_upstream_exhaustion(
    l2_model_ready: None,
    mock_a_proc: subprocess.Popen[str],
    mock_b_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    del l2_model_ready, mock_a_proc, mock_b_proc
    reset_mock(MOCK_A_URL)
    reset_mock(MOCK_B_URL)
    port = 8044
    proc = _start_gateway(_m4_l2_config(tmp_path, port, similarity_threshold=0.82), port)

    try:
        with httpx.Client(timeout=10, trust_env=False) as client:
            seed = _chat(client, port, "怎么申请退款")
            _set_behavior(MOCK_A_URL, {"fail": "error_500"})
            _set_behavior(MOCK_B_URL, {"fail": "error_500"})
            stale = _chat(client, port, "退款流程是什么")
    finally:
        _stop_gateway(proc)

    assert seed.status_code == 200, seed.text
    assert stale.status_code == 200, stale.text
    assert stale.headers["X-TideGate-Cache"] == "hit-semantic"
    assert stale.headers["X-TideGate-Route"] == "cache"
    assert "X-TideGate-Degraded" not in stale.headers


@pytest.mark.integration
def test_l2_operating_point_is_tenant_selected(
    l2_model_ready: None,
    mock_a_proc: subprocess.Popen[str],
    mock_b_proc: subprocess.Popen[str],
    tmp_path: Path,
) -> None:
    del l2_model_ready, mock_a_proc, mock_b_proc
    reset_mock(MOCK_A_URL)
    reset_mock(MOCK_B_URL)
    port = 8046
    proc = _start_gateway(_m4_l2_operating_point_config(tmp_path, port), port)

    try:
        with httpx.Client(timeout=10, trust_env=False) as client:
            aggressive_seed = _chat(
                client,
                port,
                "连续包月怎么关",
                api_key=AGGRESSIVE_API_KEY,
            )
            before_aggressive = _started_total()
            aggressive_hit = _chat(
                client,
                port,
                "关闭连续订阅入口在哪",
                api_key=AGGRESSIVE_API_KEY,
            )
            after_aggressive = _started_total()

            conservative_seed = _chat(
                client,
                port,
                "连续包月怎么关",
                api_key=CONSERVATIVE_API_KEY,
            )
            before_conservative = _started_total()
            conservative_miss = _chat(
                client,
                port,
                "关闭连续订阅入口在哪",
                api_key=CONSERVATIVE_API_KEY,
            )
            after_conservative = _started_total()
    finally:
        _stop_gateway(proc)

    assert aggressive_seed.status_code == 200, aggressive_seed.text
    assert aggressive_hit.status_code == 200, aggressive_hit.text
    assert aggressive_hit.headers["X-TideGate-Cache"] == "hit-semantic"
    assert after_aggressive == before_aggressive
    assert conservative_seed.status_code == 200, conservative_seed.text
    assert conservative_miss.status_code == 200, conservative_miss.text
    assert conservative_miss.headers["X-TideGate-Cache"] == "miss"
    assert after_conservative == before_conservative + 1


def _chat(
    client: httpx.Client,
    port: int,
    content: str,
    *,
    api_key: str = API_KEY,
    prompt_version: str = "default",
) -> httpx.Response:
    return _chat_messages(
        client,
        port,
        [{"role": "user", "content": content}],
        api_key=api_key,
        prompt_version=prompt_version,
    )


def _chat_messages(
    client: httpx.Client,
    port: int,
    messages: list[dict[str, str]],
    *,
    api_key: str = API_KEY,
    prompt_version: str = "default",
) -> httpx.Response:
    return client.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "X-Prompt-Version": prompt_version,
        },
        json={"model": "chat-large", "messages": messages, "temperature": 0.1},
    )


def _m4_l2_config(
    tmp_path: Path,
    port: int,
    *,
    add_other_tenant: bool = False,
    similarity_threshold: float = 0.75,
) -> Path:
    raw = yaml.safe_load(Path("tests/fixtures/gateway-test.yaml").read_text(encoding="utf-8"))
    raw["server"]["port"] = port
    raw["cache"]["l2"]["similarity_threshold"] = similarity_threshold
    raw["cache"]["l2"]["operating_points"] = [
        {"name": "balanced", "tau": -10.0, "expected_fpr": 1.0, "expected_recall": 1.0}
    ]
    raw["cache"]["l2"]["recall_threshold"] = min(similarity_threshold, 0.75)
    raw["cache"]["l2"]["recall_top_k"] = 3
    raw["cache"]["l2"]["stale_threshold_delta"] = 0.04
    raw["cache"]["l2"]["embed_pool_workers"] = 1
    raw["cache"]["replay_interval_ms"] = 1
    raw["tenants"][0]["cache"] = {"l1": True, "l2": True, "l2_operating_point": "balanced"}
    if add_other_tenant:
        raw["tenants"].append(
            {
                "id": "other",
                "api_key_sha256": hashlib.sha256(OTHER_API_KEY.encode("utf-8")).hexdigest(),
                "plan": "free",
                "policy": "default",
                "cache": {"l1": True, "l2": True, "l2_operating_point": "balanced"},
            }
        )
    path = tmp_path / f"gateway-m4-l2-{port}.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def _m4_l2_operating_point_config(tmp_path: Path, port: int) -> Path:
    raw = yaml.safe_load(Path("tests/fixtures/gateway-test.yaml").read_text(encoding="utf-8"))
    raw["server"]["port"] = port
    raw["cache"]["l2"]["operating_points"] = [
        {"name": "conservative", "tau": 10.0, "expected_fpr": 0.01, "expected_recall": 0.02},
        {"name": "aggressive", "tau": -10.0, "expected_fpr": 0.05, "expected_recall": 0.25},
    ]
    raw["cache"]["l2"]["recall_threshold"] = 0.50
    raw["cache"]["l2"]["recall_top_k"] = 3
    raw["cache"]["l2"]["stale_threshold_delta"] = 0.04
    raw["cache"]["l2"]["embed_pool_workers"] = 1
    raw["cache"]["replay_interval_ms"] = 1
    raw["tenants"][0]["cache"] = {"l1": True, "l2": False}
    raw["tenants"].extend(
        [
            {
                "id": "aggressive",
                "api_key_sha256": hashlib.sha256(AGGRESSIVE_API_KEY.encode("utf-8")).hexdigest(),
                "plan": "free",
                "policy": "default",
                "cache": {"l1": True, "l2": True, "l2_operating_point": "aggressive"},
            },
            {
                "id": "conservative",
                "api_key_sha256": hashlib.sha256(CONSERVATIVE_API_KEY.encode("utf-8")).hexdigest(),
                "plan": "free",
                "policy": "default",
                "cache": {"l1": True, "l2": True, "l2_operating_point": "conservative"},
            },
        ]
    )
    path = tmp_path / f"gateway-m4-l2-op-{port}.yaml"
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
    wait_ready(f"http://127.0.0.1:{port}/healthz", proc, timeout_s=30)
    return proc


def _stop_gateway(proc: subprocess.Popen[str]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _set_behavior(url: str, behavior: dict[str, object]) -> None:
    with httpx.Client(timeout=2, trust_env=False) as client:
        response = client.post(f"{url}/__behavior", json=behavior)
    assert response.status_code == 200, response.text


def _started_total() -> int:
    return stats(MOCK_A_URL)["started"] + stats(MOCK_B_URL)["started"]
