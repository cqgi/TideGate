# TideGate

TideGate is an OpenAI-compatible LLM traffic gateway for experimenting with provider failover, quota accounting, semantic cache policy, hedged streaming, cascade routing, and usage ledgers.

## Quick Start

```bash
make up
uv run --extra dev --extra test python -m mock_provider --host 127.0.0.1 --port 9001
uv run --extra dev --extra test python -m mock_provider --host 127.0.0.1 --port 9002
TIDEGATE_ADMIN_TOKEN=dev-admin MOCK_A_KEY=mock-key MOCK_B_KEY=mock-key \
  TIDEGATE_PG_DSN=postgresql://tidegate:tidegate@127.0.0.1:5432/tidegate \
  uv run --extra dev --extra test python -m tidegate --config config/gateway.yaml
```

## Verification

```bash
make check
make test
make up && uv run --extra dev --extra test pytest -m integration tests/integration && make down
```

## What Is Implemented

- OpenAI-compatible `/v1/chat/completions`, streaming SSE replay, `/v1/models`, `/metrics`.
- Per-tenant quota reservation and exact-once settlement through Redis Lua scripts.
- Provider routing with P2C selection, local circuit breakers, fallback, smaller-model degradation, and stale-cache degradation.
- L1 exact cache, L2 Redis Stack semantic cache, tenant-selected calibrated operating points, cache feedback eviction.
- Streaming hedge requests with budget limits and loser cancellation.
- Non-stream cascade routing using draft model `mean_logprob`.
- PostgreSQL usage ledger with batched idempotent inserts and shutdown drain.

## Architecture And Reports

- Architecture: `docs/architecture.md`
- Benchmark report: `out/benchmark.md`

Latest local benchmark:

| metric | value |
|---|---:|
| Gateway TTFT P99 | 102.622 ms |
| Gateway E2E P99 | 196.265 ms |
| Peak inflight | 56 |
