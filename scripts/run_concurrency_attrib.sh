#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

UV="${UV:-uv}"
PYTHON_BIN="$("$UV" run --extra dev --extra test python -c 'import sys; print(sys.executable)')"
OUT_DIR="$ROOT/out"
mkdir -p "$OUT_DIR"

MOCK_A_LOG="$OUT_DIR/mock-a-attrib.log"
MOCK_B_LOG="$OUT_DIR/mock-b-attrib.log"
GATEWAY_LOG="$OUT_DIR/gateway-attrib.log"
LOADGEN_STDOUT="$OUT_DIR/concurrency-attrib.stdout"
LOADGEN_STDERR="$OUT_DIR/concurrency-attrib.stderr"
LOADGEN_JSON="$OUT_DIR/concurrency-attrib.json"
SAMPLES_JSON="$OUT_DIR/resource-samples.json"
REPORT_MD="$OUT_DIR/attribution.md"

mock_a_pid=""
mock_b_pid=""
gateway_pid=""
loadgen_pid=""
sampler_pid=""

cleanup() {
  for pid in "$sampler_pid" "$loadgen_pid" "$gateway_pid" "$mock_a_pid" "$mock_b_pid"; do
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "$sampler_pid" "$loadgen_pid" "$gateway_pid" "$mock_a_pid" "$mock_b_pid"; do
    if [[ -n "${pid:-}" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

check_port_free() {
  local port="$1"
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "port $port is already in use" >&2
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >&2 || true
    exit 1
  fi
}

wait_http() {
  local url="$1"
  local timeout_s="${2:-15}"
  "$PYTHON_BIN" - "$url" "$timeout_s" <<'PY'
import sys
import time
import httpx

url = sys.argv[1]
deadline = time.monotonic() + float(sys.argv[2])
with httpx.Client(timeout=0.5, trust_env=False) as client:
    while time.monotonic() < deadline:
        try:
            response = client.get(url)
            if response.status_code < 500:
                raise SystemExit(0)
        except httpx.HTTPError:
            time.sleep(0.1)
raise SystemExit(f"service did not become ready: {url}")
PY
}

wait_port() {
  local host="$1"
  local port="$2"
  local timeout_s="${3:-15}"
  "$PYTHON_BIN" - "$host" "$port" "$timeout_s" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
deadline = time.monotonic() + float(sys.argv[3])
while time.monotonic() < deadline:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        try:
            sock.connect((host, port))
            raise SystemExit(0)
        except OSError:
            time.sleep(0.2)
raise SystemExit(f"port did not become ready: {host}:{port}")
PY
}

port_open() {
  local port="$1"
  (echo >/dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1
}

pg_ready() {
  local dsn="$1"
  "$PYTHON_BIN" - "$dsn" <<'PY' >/dev/null 2>&1
import asyncio
import sys

import asyncpg


async def main() -> None:
    conn = await asyncpg.connect(sys.argv[1], timeout=2)
    await conn.close()


asyncio.run(main())
PY
}

wait_pg_ready() {
  local dsn="$1"
  local timeout_s="${2:-30}"
  local deadline=$((SECONDS + timeout_s))
  while [[ "$SECONDS" -lt "$deadline" ]]; do
    if pg_ready "$dsn"; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

ulimit_before="$(ulimit -n)"
if [[ "$ulimit_before" -lt 16384 ]]; then
  ulimit -n 65536 2>/dev/null || ulimit -n 16384 2>/dev/null || true
fi
ulimit_after="$(ulimit -n)"
if [[ "$ulimit_after" -lt 16384 ]]; then
  echo "ulimit -n is $ulimit_after, expected >= 16384" >&2
  exit 1
fi
echo "ulimit -n before=$ulimit_before after=$ulimit_after"

check_port_free 8000
check_port_free 9001
check_port_free 9002

pg_port="${TIDEGATE_PG_PORT:-5432}"
pg_dsn="${TIDEGATE_PG_DSN:-postgresql://tidegate:tidegate@127.0.0.1:${pg_port}/tidegate}"
if ! (port_open 6379 && pg_ready "$pg_dsn"); then
  if [[ -z "${TIDEGATE_PG_PORT:-}" ]] && port_open 5432 && ! pg_ready "$pg_dsn"; then
    pg_port=15432
    pg_dsn="postgresql://tidegate:tidegate@127.0.0.1:${pg_port}/tidegate"
    echo "port 5432 is occupied by a non-TideGate Postgres; using TIDEGATE_PG_PORT=$pg_port"
  fi
  TIDEGATE_PG_PORT="$pg_port" make up
fi
wait_port 127.0.0.1 6379 30
wait_port 127.0.0.1 "$pg_port" 30
if ! wait_pg_ready "$pg_dsn" 30; then
  echo "Postgres is reachable on $pg_port but TideGate DSN authentication failed: $pg_dsn" >&2
  exit 1
fi

export TIDEGATE_ADMIN_TOKEN="${TIDEGATE_ADMIN_TOKEN:-dev-admin}"
export MOCK_A_KEY="${MOCK_A_KEY:-mock-key}"
export MOCK_B_KEY="${MOCK_B_KEY:-mock-key}"
export TIDEGATE_PG_DSN="$pg_dsn"
export PYTHONPATH="$ROOT/src:$ROOT"

"$PYTHON_BIN" -m mock_provider --host 127.0.0.1 --port 9001 >"$MOCK_A_LOG" 2>&1 &
mock_a_pid=$!
"$PYTHON_BIN" -m mock_provider --host 127.0.0.1 --port 9002 >"$MOCK_B_LOG" 2>&1 &
mock_b_pid=$!
wait_http "http://127.0.0.1:9001/__stats" 15
wait_http "http://127.0.0.1:9002/__stats" 15

"$PYTHON_BIN" -m tidegate --config out/bench-default.yaml >"$GATEWAY_LOG" 2>&1 &
gateway_pid=$!
wait_http "http://127.0.0.1:8000/healthz" 20

"$PYTHON_BIN" scripts/loadgen.py \
  --scenario concurrency \
  --rps "${TIDEGATE_BENCH_RPS:-115}" \
  --duration "${TIDEGATE_BENCH_DURATION:-31}" \
  --stream-ratio 1.0 \
  --mock-tpot-ms "${TIDEGATE_BENCH_TPOT_MS:-300}" \
  --mock-output-tokens "${TIDEGATE_BENCH_OUTPUT_TOKENS:-100}" \
  --gateway-pid "$gateway_pid" \
  --max-connections "${TIDEGATE_BENCH_MAX_CONNECTIONS:-10000}" \
  --output "$LOADGEN_JSON" >"$LOADGEN_STDOUT" 2>"$LOADGEN_STDERR" &
loadgen_pid=$!

"$PYTHON_BIN" scripts/resource_sampler.py \
  --pids "gateway=$gateway_pid" \
  --pids "mock-a=$mock_a_pid" \
  --pids "mock-b=$mock_b_pid" \
  --pids "loadgen=$loadgen_pid" \
  --interval "${TIDEGATE_ATTRIB_SAMPLE_INTERVAL:-1.0}" \
  --out "$SAMPLES_JSON" &
sampler_pid=$!

wait "$loadgen_pid"
loadgen_pid=""

if [[ -n "$sampler_pid" ]] && kill -0 "$sampler_pid" 2>/dev/null; then
  kill -TERM "$sampler_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true
fi
sampler_pid=""

"$PYTHON_BIN" scripts/make_attrib_report.py \
  --current "$LOADGEN_JSON" \
  --samples "$SAMPLES_JSON" \
  --history out/concurrency.json \
  --output "$REPORT_MD" >/dev/null

echo "wrote $LOADGEN_JSON"
echo "wrote $SAMPLES_JSON"
echo "wrote $REPORT_MD"
