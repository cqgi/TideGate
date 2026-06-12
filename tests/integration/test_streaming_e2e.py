from __future__ import annotations

import json
import socket
import struct
import subprocess
import time

import httpx
import pytest
from openai import OpenAI

from tests.integration.conftest import API_KEY, BASE_URL, MOCK_A_URL, MOCK_B_URL, reset_mock


@pytest.mark.integration
def test_stream_and_non_stream_match(gateway_proc: subprocess.Popen[str]) -> None:
    """SPEC-M0-5."""
    del gateway_proc
    directive = {"ttft_ms": 10, "tpot_ms": 1, "output_tokens": 5}
    headers = {"x-mock-directive": json.dumps(directive)}
    client = OpenAI(
        api_key=API_KEY,
        base_url=f"{BASE_URL}/v1",
        default_headers=headers,
        http_client=httpx.Client(trust_env=False),
    )

    non_stream = client.chat.completions.create(
        model="chat-large",
        messages=[{"role": "user", "content": "hi"}],
    )
    stream = client.chat.completions.create(
        model="chat-large",
        stream=True,
        messages=[{"role": "user", "content": "hi"}],
    )
    streamed = "".join(chunk.choices[0].delta.content or "" for chunk in stream)

    assert non_stream.choices[0].message.content == streamed


@pytest.mark.integration
def test_disconnect_aborts_upstream(gateway_proc: subprocess.Popen[str]) -> None:
    """SPEC-M0-5."""
    del gateway_proc
    reset_mock(MOCK_A_URL)
    reset_mock(MOCK_B_URL)
    directive = {"ttft_ms": 10, "tpot_ms": 80, "output_tokens": 50}
    body = json.dumps(
        {
            "model": "chat-large",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    request_bytes = (
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: 127.0.0.1:8000\r\n"
        f"Authorization: Bearer {API_KEY}\r\n"
        f"x-mock-directive: {json.dumps(directive)}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body.encode())}\r\n"
        "Connection: close\r\n"
        "\r\n"
        f"{body}"
    ).encode()
    with socket.create_connection(("127.0.0.1", 8000), timeout=5) as sock:
        sock.sendall(request_bytes)
        sock.settimeout(5)
        data = b""
        while data.count(b"data:") < 3:
            data += sock.recv(1024)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        sock.shutdown(socket.SHUT_RDWR)

    deadline = time.monotonic() + 2
    with httpx.Client(timeout=2, trust_env=False) as client:
        while time.monotonic() < deadline:
            stats_a = client.get(f"{MOCK_A_URL}/__stats").json()
            stats_b = client.get(f"{MOCK_B_URL}/__stats").json()
            if stats_a["aborted"] + stats_b["aborted"] >= 1:
                return
            time.sleep(0.1)
    raise AssertionError("mock provider did not record upstream abort")


@pytest.mark.integration
def test_error_shapes(gateway_proc: subprocess.Popen[str]) -> None:
    """SPEC-M0-3."""
    del gateway_proc
    with httpx.Client(timeout=2, trust_env=False) as client:
        unauthorized = client.post(
            f"{BASE_URL}/v1/chat/completions",
            json={"model": "chat-large", "messages": [{"role": "user", "content": "hi"}]},
        )
        not_found = client.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": "missing", "messages": [{"role": "user", "content": "hi"}]},
        )
        invalid = client.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": "chat-large", "messages": [{"role": "bad", "content": "hi"}]},
        )
        bad_json = client.post(
            f"{BASE_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            content="{bad json",
        )

    assert unauthorized.status_code == 401
    assert unauthorized.json()["error"]["type"] == "authentication_error"
    assert not_found.status_code == 404
    assert not_found.json()["error"]["type"] == "invalid_request_error"
    assert invalid.status_code == 422
    assert invalid.json()["error"]["type"] == "invalid_request_error"
    assert bad_json.status_code == 422
    assert bad_json.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.integration
def test_heartbeat(gateway_proc: subprocess.Popen[str]) -> None:
    """SPEC-M0-5."""
    del gateway_proc
    directive = {"ttft_ms": 2000, "tpot_ms": 1, "output_tokens": 1}
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "x-mock-directive": json.dumps(directive),
    }
    with (
        httpx.Client(timeout=5, trust_env=False) as client,
        client.stream(
            "POST",
            f"{BASE_URL}/v1/chat/completions",
            headers=headers,
            json={
                "model": "chat-large",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as response,
    ):
        for line in response.iter_lines():
            if line == ": ping":
                return
    raise AssertionError("heartbeat was not emitted before first chunk")
