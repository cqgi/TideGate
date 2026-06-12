from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

import httpx


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8000")
    parser.add_argument("--mock-a-url", default="http://127.0.0.1:9001")
    parser.add_argument("--mock-b-url", default="http://127.0.0.1:9002")
    parser.add_argument("--api-key", default="<demo key>")
    parser.add_argument("--rps", type=float, default=30.0)
    parser.add_argument("--inject-after-s", type=float, default=10.0)
    parser.add_argument("--recover-after-s", type=float, default=40.0)
    parser.add_argument("--duration-s", type=float, default=50.0)
    parser.add_argument("--report", default="out/chaos_failover.json")
    args = parser.parse_args(argv)

    interval_s = 1.0 / max(1.0, args.rps)
    started = time.monotonic()
    injected_at: float | None = None
    recovered_at: float | None = None
    first_b_streak_at: float | None = None
    first_a_after_recovery_at: float | None = None
    consecutive_b = 0
    total = 0
    ok = 0
    during_total = 0
    during_ok = 0
    routes: dict[str, int] = {}

    _post(args.mock_a_url, "/__reset", {})
    _post(args.mock_b_url, "/__reset", {})
    with httpx.Client(timeout=5, trust_env=False) as client:
        while time.monotonic() - started < args.duration_s:
            elapsed = time.monotonic() - started
            if injected_at is None and elapsed >= args.inject_after_s:
                _post(args.mock_a_url, "/__behavior", {"fail": "error_500"})
                injected_at = time.monotonic()
                consecutive_b = 0
                first_b_streak_at = None
            if recovered_at is None and elapsed >= args.recover_after_s:
                _post(args.mock_a_url, "/__behavior", {"fail": "none", "ttft_ms": 10})
                recovered_at = time.monotonic()
                consecutive_b = 0

            response = _chat(client, args.gateway_url, args.api_key)
            total += 1
            if response.status_code == 200:
                ok += 1
            if injected_at is not None and recovered_at is None:
                during_total += 1
                if response.status_code == 200:
                    during_ok += 1
            route = response.headers.get("X-TideGate-Route", "none")
            routes[route] = routes.get(route, 0) + 1
            if injected_at is not None and recovered_at is None and route.startswith("mock-b/"):
                consecutive_b += 1
                if consecutive_b >= 10 and first_b_streak_at is None:
                    first_b_streak_at = time.monotonic()
            elif injected_at is not None and recovered_at is None:
                consecutive_b = 0
            if recovered_at is not None and route.startswith("mock-a/"):
                first_a_after_recovery_at = first_a_after_recovery_at or time.monotonic()
            time.sleep(interval_s)

    report = {
        "total_requests": total,
        "success_rate": ok / total if total else 0.0,
        "during_failure_success_rate": during_ok / during_total if during_total else 0.0,
        "failover_seconds": (
            None
            if injected_at is None or first_b_streak_at is None
            else first_b_streak_at - injected_at
        ),
        "recovery_seconds": (
            None
            if recovered_at is None or first_a_after_recovery_at is None
            else first_a_after_recovery_at - recovered_at
        ),
        "routes": routes,
    }
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


def _chat(client: httpx.Client, gateway_url: str, api_key: str) -> httpx.Response:
    return client.post(
        f"{gateway_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": "chat-large", "messages": [{"role": "user", "content": "hi"}]},
    )


def _post(base_url: str, path: str, payload: dict[str, object]) -> None:
    with httpx.Client(timeout=2, trust_env=False) as client:
        response = client.post(f"{base_url}{path}", json=payload)
    response.raise_for_status()


if __name__ == "__main__":
    main()
