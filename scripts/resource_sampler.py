from __future__ import annotations

import argparse
import json
import platform
import resource
import signal
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import psutil

_STOP = False


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pids", action="append", default=[], metavar="NAME=PID")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--out", default="out/resource-samples.json")
    args = parser.parse_args(argv)

    targets = _parse_targets(args.pids)
    processes = {name: _process(pid) for name, pid in targets.items()}
    started_wall = time.time()
    started = time.monotonic()
    payload: dict[str, Any] = {
        "meta": {
            "started_at": _iso_timestamp(started_wall),
            "interval_s": args.interval,
            "platform": platform.platform(),
            "logical_cpus": psutil.cpu_count(logical=True),
            "physical_cpus": psutil.cpu_count(logical=False),
            "memory_total_bytes": psutil.virtual_memory().total,
            "ulimit_nofile_soft": resource.getrlimit(resource.RLIMIT_NOFILE)[0],
            "ulimit_nofile_hard": resource.getrlimit(resource.RLIMIT_NOFILE)[1],
            "targets": targets,
        },
        "samples": [],
    }

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    psutil.cpu_percent(interval=None, percpu=True)
    psutil.cpu_percent(interval=None)
    for proc in processes.values():
        if proc is not None:
            _safe_call(proc.cpu_percent, None)

    try:
        while not _STOP:
            time.sleep(args.interval)
            payload["samples"].append(_sample(started, processes))
            if all(proc is None or not proc.is_running() for proc in processes.values()):
                break
    finally:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _parse_targets(raw_targets: Sequence[str]) -> dict[str, int]:
    targets: dict[str, int] = {}
    for raw in raw_targets:
        if "=" not in raw:
            raise SystemExit(f"invalid --pids value, expected NAME=PID: {raw}")
        name, pid_raw = raw.split("=", maxsplit=1)
        try:
            targets[name] = int(pid_raw)
        except ValueError as exc:
            raise SystemExit(f"invalid pid for {name}: {pid_raw}") from exc
    if not targets:
        raise SystemExit("at least one --pids NAME=PID target is required")
    return targets


def _process(pid: int) -> psutil.Process | None:
    try:
        return psutil.Process(pid)
    except psutil.Error:
        return None


def _sample(started: float, processes: dict[str, psutil.Process | None]) -> dict[str, Any]:
    return {
        "time_s": round(time.monotonic() - started, 3),
        "timestamp": _iso_timestamp(time.time()),
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "cpu_percent_per_core": psutil.cpu_percent(interval=None, percpu=True),
            "memory": _memory_snapshot(),
            "net_io_counters": _net_snapshot(),
        },
        "processes": {name: _process_snapshot(proc) for name, proc in processes.items()},
    }


def _memory_snapshot() -> dict[str, float | int]:
    memory = psutil.virtual_memory()
    return {
        "total": memory.total,
        "available": memory.available,
        "used": memory.used,
        "percent": memory.percent,
    }


def _net_snapshot() -> dict[str, dict[str, int]]:
    counters = psutil.net_io_counters(pernic=True)
    return {
        name: {
            "bytes_sent": stat.bytes_sent,
            "bytes_recv": stat.bytes_recv,
            "packets_sent": stat.packets_sent,
            "packets_recv": stat.packets_recv,
            "errin": stat.errin,
            "errout": stat.errout,
            "dropin": stat.dropin,
            "dropout": stat.dropout,
        }
        for name, stat in counters.items()
    }


def _process_snapshot(proc: psutil.Process | None) -> dict[str, object]:
    if proc is None:
        return {"running": False, "error": "process_not_found"}
    try:
        running = proc.is_running()
        if not running:
            return {"pid": proc.pid, "running": False}
        memory = proc.memory_info()
        return {
            "pid": proc.pid,
            "running": True,
            "cpu_percent": proc.cpu_percent(interval=None),
            "rss_bytes": memory.rss,
            "rss_mb": round(memory.rss / 1024 / 1024, 3),
            "num_fds": _safe_call(proc.num_fds),
            "num_threads": proc.num_threads(),
            "status": proc.status(),
        }
    except psutil.Error as exc:
        return {"pid": proc.pid, "running": False, "error": type(exc).__name__}


def _safe_call(func: object, *args: object) -> object | None:
    try:
        if not callable(func):
            return None
        return cast(Callable[..., object], func)(*args)
    except (OSError, psutil.Error, PermissionError):
        return None


def _handle_stop(_signum: int, _frame: object) -> None:
    global _STOP
    _STOP = True


def _iso_timestamp(value: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))


if __name__ == "__main__":
    main()
