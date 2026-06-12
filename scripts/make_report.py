from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--baseline")
    parser.add_argument("--gateway")
    parser.add_argument("--concurrency")
    parser.add_argument("--cache-hit")
    parser.add_argument("--hedge-off")
    parser.add_argument("--hedge-on")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    runs = _load_runs(args)
    report = _render_report(runs)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
    print(report, end="")


def _load_runs(args: argparse.Namespace) -> list[tuple[str, dict[str, object]]]:
    named = [
        ("baseline", args.baseline),
        ("gateway", args.gateway),
        ("concurrency", args.concurrency),
        ("cache-hit", args.cache_hit),
        ("hedge-off", args.hedge_off),
        ("hedge-on", args.hedge_on),
    ]
    runs: list[tuple[str, dict[str, object]]] = []
    for name, path in named:
        if path is not None:
            runs.append((name, _load_json(path)))
    for path in args.runs:
        data = _load_json(path)
        name = str(data.get("scenario") or Path(path).stem)
        runs.append((name, data))
    if not runs:
        raise SystemExit("at least one benchmark JSON path is required")
    return runs


def _load_json(path: str) -> dict[str, object]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"benchmark file is not a JSON object: {path}")
    return data


def _render_report(runs: list[tuple[str, dict[str, object]]]) -> str:
    lines: list[str] = []
    lines.append("# TideGate Benchmark Report")
    lines.append("")
    lines.append(
        "| run | requests | success_rate | ttft_p50_ms | ttft_p95_ms | "
        "ttft_p99_ms | e2e_p99_ms | gateway_overhead_p99_ms | loop_lag_peak_s | "
        "peak_inflight | rss_mb |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, data in runs:
        lines.append(
            f"| {name} | {data.get('requests', '')} | {_fmt(data.get('success_rate'))} | "
            f"{_metric(data, 'ttft_ms', 'p50')} | {_metric(data, 'ttft_ms', 'p95')} | "
            f"{_metric(data, 'ttft_ms', 'p99')} | {_metric(data, 'e2e_ms', 'p99')} | "
            f"{_fmt(data.get('gateway_overhead_p99_ms'))} | "
            f"{_fmt(data.get('loop_lag_peak_s'))} | {data.get('peak_inflight', '')} | "
            f"{_fmt(data.get('gateway_rss_mb'))} |"
        )
    lines.append("")
    lines.append("## Scenario Details")
    lines.append("")
    for name, data in runs:
        lines.append(f"### {name}")
        lines.append("")
        cache_headers = json.dumps(data.get("cache_headers", {}), sort_keys=True)
        mock_directive = json.dumps(data.get("mock_directive", {}), sort_keys=True)
        lines.append(f"- Cache headers: `{cache_headers}`")
        lines.append(f"- L1 hit rate: {_fmt(data.get('cache_l1_hit_rate'))}")
        lines.append(
            f"- Cache hit TTFT P50/P95/P99: {_metric(data, 'cache_hit_ttft_ms', 'p50')} / "
            f"{_metric(data, 'cache_hit_ttft_ms', 'p95')} / "
            f"{_metric(data, 'cache_hit_ttft_ms', 'p99')} ms"
        )
        lines.append(f"- Mock directive: `{mock_directive}`")
        lines.append("")
    lines.append("## Hedge Effect")
    lines.append("")
    hedge_off = _find(runs, "hedge-off")
    hedge_on = _find(runs, "hedge-on")
    if hedge_off is not None and hedge_on is not None:
        off = _metric_value(hedge_off, "ttft_ms", "p99")
        on = _metric_value(hedge_on, "ttft_ms", "p99")
        delta = "" if off is None or on is None else _fmt(off - on)
        lines.append(f"- TTFT P99 off: {_fmt(off)} ms")
        lines.append(f"- TTFT P99 on: {_fmt(on)} ms")
        lines.append(f"- P99 improvement: {delta} ms")
    else:
        lines.append("- Hedge off/on runs were not both provided.")
    lines.append("")
    lines.append("## Report Notes")
    lines.append("")
    lines.append(
        "- TTFT absolute values reflect mock provider speed. Gateway capacity is read from "
        "baseline/gateway deltas, gateway_overhead_p99_ms, and loop_lag_peak_s."
    )
    lines.append(
        "- Concurrency target is validated by peak_inflight; loop_lag_peak_s is reported as "
        "observed instead of capped or smoothed."
    )
    lines.append("")
    return "\n".join(lines)


def _find(runs: list[tuple[str, dict[str, object]]], name: str) -> dict[str, object] | None:
    for run_name, data in runs:
        if run_name == name:
            return data
    return None


def _metric(data: dict[str, object], section: str, key: str) -> str:
    value = _metric_value(data, section, key)
    return _fmt(value)


def _metric_value(data: dict[str, object], section: str, key: str) -> float | None:
    values = data.get(section)
    if not isinstance(values, dict):
        return None
    value = values.get(key)
    if isinstance(value, int | float):
        return float(value)
    return None


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, int | float):
        return f"{float(value):.3f}"
    return str(value)


if __name__ == "__main__":
    main()
