from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import time
from collections.abc import Sequence
from urllib.parse import urljoin

import httpx


async def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--api-key", default="<demo key>")
    parser.add_argument("--rps", type=float, default=20.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--stream-ratio", type=float, default=0.5)
    parser.add_argument("--cache-hit-ratio", type=float, default=0.0)
    parser.add_argument("--mock-ttft-lognorm", default=None)
    args = parser.parse_args(argv)

    deadline = time.monotonic() + args.duration
    tasks: set[asyncio.Task[dict[str, float | int | bool]]] = set()
    all_tasks: list[asyncio.Task[dict[str, float | int | bool]]] = []
    sent = 0
    peak_inflight = 0
    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
        while time.monotonic() < deadline:
            sent += 1
            stream = random.random() < args.stream_ratio
            prompt = _prompt(sent, args.cache_hit_ratio)
            task = asyncio.create_task(_one(client, args, prompt, stream))
            tasks.add(task)
            all_tasks.append(task)
            peak_inflight = max(peak_inflight, len(tasks))
            tasks = {task for task in tasks if not task.done()}
            await asyncio.sleep(random.expovariate(args.rps))
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        gateway_metrics = await _read_gateway_metrics(client, args.url)

    rows = [row for row in results if isinstance(row, dict)]
    ttfts = [float(row["ttft_ms"]) for row in rows if row.get("ok")]
    e2e = [float(row["e2e_ms"]) for row in rows if row.get("ok")]
    output = {
        "requests": sent,
        "completed": len(rows),
        "success_rate": sum(1 for row in rows if row.get("ok")) / max(1, len(rows)),
        "ttft_ms": _percentiles(ttfts),
        "e2e_ms": _percentiles(e2e),
        "gateway_overhead_p99_ms": gateway_metrics["gateway_overhead_p99_ms"],
        "peak_inflight": peak_inflight,
        "loop_lag_peak_s": gateway_metrics["loop_lag_peak_s"],
    }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))


async def _one(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    prompt: str,
    stream: bool,
) -> dict[str, float | int | bool]:
    headers = {"Authorization": f"Bearer {args.api_key}"}
    if args.mock_ttft_lognorm:
        headers["x-mock-directive"] = json.dumps({"ttft_lognorm": args.mock_ttft_lognorm})
    body = {
        "model": "chat-large",
        "stream": stream,
        "messages": [{"role": "user", "content": prompt}],
    }
    started = time.monotonic()
    first_byte: float | None = None
    async with client.stream("POST", args.url, headers=headers, json=body) as response:
        async for chunk in response.aiter_bytes():
            if first_byte is None and chunk:
                first_byte = time.monotonic()
        ended = time.monotonic()
    return {
        "ok": response.status_code == 200,
        "ttft_ms": ((first_byte or ended) - started) * 1000,
        "e2e_ms": (ended - started) * 1000,
    }


def _prompt(index: int, cache_hit_ratio: float) -> str:
    if cache_hit_ratio > 0 and random.random() < cache_hit_ratio:
        return f"cacheable shared prompt {index % 10}"
    # DECISION: REWORK-M4-3 makes loadgen unique by default so latency numbers are not
    # accidentally dominated by L1/L2 cache hits.
    return f"unique loadgen prompt {index}-{time.monotonic_ns()}"


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "p99": None}
    values = sorted(values)
    return {
        "p50": _pct(values, 0.50),
        "p95": _pct(values, 0.95),
        "p99": _pct(values, 0.99),
    }


def _pct(values: list[float], q: float) -> float:
    index = min(len(values) - 1, math.ceil(len(values) * q) - 1)
    return round(values[index], 3)


async def _read_gateway_metrics(
    client: httpx.AsyncClient,
    chat_url: str,
) -> dict[str, float | None]:
    metrics_url = urljoin(chat_url, "/metrics")
    try:
        response = await client.get(metrics_url, timeout=2)
        if response.status_code != 200:
            return {"gateway_overhead_p99_ms": None, "loop_lag_peak_s": None}
    except httpx.HTTPError:
        return {"gateway_overhead_p99_ms": None, "loop_lag_peak_s": None}
    return {
        "gateway_overhead_p99_ms": _histogram_p99_ms(
            response.text,
            "tidegate_gateway_overhead_seconds",
        ),
        "loop_lag_peak_s": _gauge_value(response.text, "tidegate_loop_lag_seconds"),
    }


def _histogram_p99_ms(text: str, metric: str) -> float | None:
    buckets: list[tuple[float, float]] = []
    for line in text.splitlines():
        if not line.startswith(f"{metric}_bucket"):
            continue
        le_match = re.search(r'le="([^"]+)"', line)
        value_match = re.search(r"} ([0-9.eE+-]+)$", line)
        if le_match is None or value_match is None:
            continue
        le_raw = le_match.group(1)
        le = math.inf if le_raw == "+Inf" else float(le_raw)
        buckets.append((le, float(value_match.group(1))))
    if not buckets:
        return None
    buckets.sort(key=lambda item: item[0])
    total = buckets[-1][1]
    if total <= 0:
        return None
    target = total * 0.99
    previous_le = 0.0
    previous_count = 0.0
    for le, count in buckets:
        if count >= target:
            if math.isinf(le):
                return round(previous_le * 1000, 3)
            bucket_count = count - previous_count
            if bucket_count <= 0:
                return round(le * 1000, 3)
            fraction = (target - previous_count) / bucket_count
            return round((previous_le + (le - previous_le) * fraction) * 1000, 3)
        previous_le = le
        previous_count = count
    return None


def _gauge_value(text: str, metric: str) -> float | None:
    for line in text.splitlines():
        if line.startswith(f"{metric} "):
            return float(line.rsplit(" ", maxsplit=1)[1])
    return None


if __name__ == "__main__":
    asyncio.run(main())
