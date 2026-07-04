from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

MB = 1024 * 1024


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", default="out/concurrency-attrib.json")
    parser.add_argument("--samples", default="out/resource-samples.json")
    parser.add_argument("--history", default="out/concurrency.json")
    parser.add_argument("--output", default="out/attribution.md")
    args = parser.parse_args(argv)

    current = _load_json(args.current)
    samples = _load_json(args.samples)
    history = _load_json(args.history)
    report = _render_report(current, samples, history)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(report, end="")


def _load_json(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"JSON root is not an object: {path}")
    return data


def _render_report(
    current: dict[str, Any],
    samples_payload: dict[str, Any],
    history: dict[str, Any],
) -> str:
    samples = [sample for sample in samples_payload.get("samples", []) if isinstance(sample, dict)]
    resource = _resource_summary(samples, samples_payload.get("meta", {}))
    errors = current.get("errors_by_category", {})

    lines: list[str] = []
    lines.append("# TideGate Concurrency Attribution Report")
    lines.append("")
    lines.append("## Meta")
    lines.append("")
    lines.append(f"- Started at: {samples_payload.get('meta', {}).get('started_at', '')}")
    lines.append(f"- Sample interval: {_fmt(samples_payload.get('meta', {}).get('interval_s'))} s")
    lines.append(
        f"- ulimit -n: soft={samples_payload.get('meta', {}).get('ulimit_nofile_soft', '')}, "
        f"hard={samples_payload.get('meta', {}).get('ulimit_nofile_hard', '')}"
    )
    targets = json.dumps(samples_payload.get("meta", {}).get("targets", {}), sort_keys=True)
    lines.append(f"- Targets: `{targets}`")
    lines.append("")
    lines.append("## Benchmark Comparison")
    lines.append("")
    lines.append(
        "| run | requests | peak_inflight | success_rate | ttft_p50_ms | ttft_p95_ms | "
        "ttft_p99_ms | loop_lag_peak_s | gateway_overhead_p99_ms | gateway_rss_mb |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(_benchmark_row("history", history))
    lines.append(_benchmark_row("current", current))
    lines.append("")
    lines.append("## Resource Peak Summary")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| system_cpu_mean_percent | {_fmt(resource['system_cpu_mean_percent'])} |")
    lines.append(f"| system_cpu_peak_percent | {_fmt(resource['system_cpu_peak_percent'])} |")
    busiest_core = _fmt(resource["busiest_single_core_peak_percent"])
    lines.append(f"| busiest_single_core_peak_percent | {busiest_core} |")
    lines.append(f"| gateway_cpu_peak_percent | {_fmt(resource['gateway_cpu_peak_percent'])} |")
    lines.append(f"| gateway_cpu_mean_percent | {_fmt(resource['gateway_cpu_mean_percent'])} |")
    lines.append(f"| gateway_rss_peak_mb | {_fmt(resource['gateway_rss_peak_mb'])} |")
    lines.append(f"| gateway_fd_peak | {_fmt(resource['gateway_fd_peak'])} |")
    lines.append(
        f"| loopback_throughput_peak_mib_s | {_fmt(resource['loopback_throughput_peak_mib_s'])} |"
    )
    lines.append(f"| top_process_by_cpu_peak | {resource['top_process_by_cpu_peak']} |")
    lines.append("")
    lines.append("## Bottleneck Attribution")
    lines.append("")
    lines.append(_conclusion(resource, samples_payload.get("meta", {})))
    lines.append("")
    lines.append("## Failure Attribution")
    lines.append("")
    lines.append("| category | count | samples |")
    lines.append("|---|---:|---|")
    if isinstance(errors, dict) and errors:
        for category, value in sorted(errors.items()):
            if not isinstance(value, dict):
                continue
            samples_text = "<br>".join(str(item) for item in value.get("samples", []))
            lines.append(f"| {category} | {value.get('count', 0)} | {samples_text} |")
    else:
        lines.append("| none | 0 |  |")
    lines.append("")
    lines.append("## Honesty Notes")
    lines.append("")
    lines.append(
        "- Gateway, mock providers, Redis/Postgres, and loadgen ran on the same machine, "
        "so the absolute numbers include local process contention."
    )
    lines.append(
        "- Mock provider and loadgen CPU are part of the measured environment; if either is "
        "hotter than the gateway, the client/provider side may be the earlier bottleneck."
    )
    lines.append(
        "- Loopback throughput is reported as an observed MiB/s value. macOS loopback has no "
        "physical NIC capacity exposed here, so this report does not convert it into percent "
        "of link capacity."
    )
    lines.append(
        "- These results are for deterministic mock-provider behavior and should not be read as "
        "production provider capacity."
    )
    lines.append("")
    return "\n".join(lines)


def _benchmark_row(name: str, data: dict[str, Any]) -> str:
    return (
        f"| {name} | {data.get('requests', '')} | {data.get('peak_inflight', '')} | "
        f"{_fmt(data.get('success_rate'))} | "
        f"{_metric(data, 'ttft_ms', 'p50')} | {_metric(data, 'ttft_ms', 'p95')} | "
        f"{_metric(data, 'ttft_ms', 'p99')} | {_fmt(data.get('loop_lag_peak_s'))} | "
        f"{_fmt(data.get('gateway_overhead_p99_ms'))} | {_fmt(data.get('gateway_rss_mb'))} |"
    )


def _resource_summary(samples: list[dict[str, Any]], meta: object) -> dict[str, object]:
    cpu_avgs: list[float] = []
    core_peaks: list[float] = []
    gateway_cpu: list[float] = []
    gateway_rss: list[float] = []
    gateway_fds: list[float] = []
    process_cpu_peaks: dict[str, float] = {}
    loopback_rates: list[float] = []
    previous_sample: dict[str, Any] | None = None
    for sample in samples:
        system = sample.get("system", {})
        if isinstance(system, dict):
            cpu = _number(system.get("cpu_percent"))
            if cpu is not None:
                cpu_avgs.append(cpu)
            per_core = _number_list(system.get("cpu_percent_per_core"))
            if per_core:
                core_peaks.append(max(per_core))
            loopback = _loopback_throughput_mib_s(previous_sample, sample)
            if loopback is not None:
                loopback_rates.append(loopback)
        processes = sample.get("processes", {})
        if isinstance(processes, dict):
            for name, raw_proc in processes.items():
                if not isinstance(raw_proc, dict):
                    continue
                cpu = _number(raw_proc.get("cpu_percent"))
                if cpu is not None:
                    process_cpu_peaks[name] = max(cpu, process_cpu_peaks.get(name, 0.0))
                    if name == "gateway":
                        gateway_cpu.append(cpu)
                if name == "gateway":
                    rss = _number(raw_proc.get("rss_mb"))
                    if rss is not None:
                        gateway_rss.append(rss)
                    fds = _number(raw_proc.get("num_fds"))
                    if fds is not None:
                        gateway_fds.append(fds)
        previous_sample = sample
    top_process = ""
    if process_cpu_peaks:
        process, value = max(process_cpu_peaks.items(), key=lambda item: item[1])
        top_process = f"{process} ({value:.3f}%)"
    del meta
    return {
        "system_cpu_mean_percent": _mean(cpu_avgs),
        "system_cpu_peak_percent": max(cpu_avgs, default=None),
        "busiest_single_core_peak_percent": max(core_peaks, default=None),
        "gateway_cpu_peak_percent": max(gateway_cpu, default=None),
        "gateway_cpu_mean_percent": _mean(gateway_cpu),
        "gateway_rss_peak_mb": max(gateway_rss, default=None),
        "gateway_fd_peak": max(gateway_fds, default=None),
        "loopback_throughput_peak_mib_s": max(loopback_rates, default=None),
        "top_process_by_cpu_peak": top_process,
    }


def _loopback_throughput_mib_s(
    previous_sample: dict[str, Any] | None,
    sample: dict[str, Any],
) -> float | None:
    if previous_sample is None:
        return None
    previous_time = _number(previous_sample.get("time_s"))
    current_time = _number(sample.get("time_s"))
    if previous_time is None or current_time is None or current_time <= previous_time:
        return None
    previous = _loopback_bytes(previous_sample)
    current = _loopback_bytes(sample)
    if previous is None or current is None or current < previous:
        return None
    return (current - previous) / (current_time - previous_time) / MB


def _loopback_bytes(sample: dict[str, Any]) -> int | None:
    system = sample.get("system", {})
    if not isinstance(system, dict):
        return None
    counters = system.get("net_io_counters", {})
    if not isinstance(counters, dict):
        return None
    candidates = [name for name in counters if name == "lo0" or str(name).startswith("lo")]
    if not candidates:
        return None
    total = 0
    for name in candidates:
        stat = counters.get(name, {})
        if not isinstance(stat, dict):
            continue
        sent = _number(stat.get("bytes_sent")) or 0
        recv = _number(stat.get("bytes_recv")) or 0
        total += int(sent + recv)
    return total


def _conclusion(resource: dict[str, object], meta: object) -> str:
    gateway_cpu_peak = _number(resource.get("gateway_cpu_peak_percent")) or 0.0
    gateway_fd_peak = _number(resource.get("gateway_fd_peak")) or 0.0
    nofile = 0.0
    if isinstance(meta, dict):
        nofile = _number(meta.get("ulimit_nofile_soft")) or 0.0
    cpu_sentence = (
        "Gateway CPU did not approach a full single core."
        if gateway_cpu_peak < 90.0
        else "Gateway CPU approached or exceeded a full single core."
    )
    fd_sentence = (
        f" Gateway fd peak was {gateway_fd_peak:.0f} against ulimit {nofile:.0f}."
        if nofile > 0
        else " Gateway fd headroom could not be computed because ulimit was unavailable."
    )
    return (
        f"{cpu_sentence} Gateway CPU peak was {_fmt(gateway_cpu_peak)}%, while the busiest "
        f"observed target process was {resource.get('top_process_by_cpu_peak') or 'unknown'}. "
        f"System CPU peak was {_fmt(resource.get('system_cpu_peak_percent'))}% and busiest "
        f"single-core peak was {_fmt(resource.get('busiest_single_core_peak_percent'))}%. "
        f"Gateway RSS peaked at {_fmt(resource.get('gateway_rss_peak_mb'))} MiB and loopback "
        f"throughput peaked at {_fmt(resource.get('loopback_throughput_peak_mib_s'))} MiB/s."
        f"{fd_sentence}"
    )


def _metric(data: dict[str, Any], section: str, key: str) -> str:
    values = data.get(section)
    if not isinstance(values, dict):
        return ""
    return _fmt(values.get(key))


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _number_list(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    return [number for item in value if (number := _number(item)) is not None]


def _mean(values: Iterable[float]) -> float | None:
    collected = list(values)
    if not collected:
        return None
    return statistics.fmean(collected)


def _fmt(value: object) -> str:
    number = _number(value)
    if number is not None:
        return f"{number:.3f}"
    if value is None:
        return ""
    return str(value)


if __name__ == "__main__":
    main()
