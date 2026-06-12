from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("gateway")
    args = parser.parse_args(argv)

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    gateway = json.loads(Path(args.gateway).read_text(encoding="utf-8"))
    print("# TideGate Benchmark Report")
    print()
    print("| run | requests | success_rate | ttft_p95_ms | ttft_p99_ms | e2e_p99_ms |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, data in [("baseline", baseline), ("gateway", gateway)]:
        print(
            f"| {name} | {data['requests']} | {data['success_rate']:.3f} | "
            f"{_metric(data, 'ttft_ms', 'p95')} | {_metric(data, 'ttft_ms', 'p99')} | "
            f"{_metric(data, 'e2e_ms', 'p99')} |"
        )
    print()
    print("## Resume Numbers")
    print()
    print(f"- Gateway TTFT P99: {_metric(gateway, 'ttft_ms', 'p99')} ms")
    print(f"- Gateway E2E P99: {_metric(gateway, 'e2e_ms', 'p99')} ms")
    print(f"- Peak inflight: {gateway.get('peak_inflight')}")


def _metric(data: dict[str, object], section: str, key: str) -> object:
    values = data.get(section)
    if not isinstance(values, dict):
        return ""
    return values.get(key, "")


if __name__ == "__main__":
    main()
