# TideGate

TideGate is an OpenAI-compatible traffic gateway I built to exercise the parts of LLM serving that become painful once several apps share the same model spend. Directly wiring every service to a provider makes cost hard to see, failover hard to test, repeated prompts expensive, and streaming behavior easy to get wrong. This repo keeps the scope narrow: put one gateway between the client and a small provider pool, then make quota, cache, routing, and tail latency behavior explicit.

```
OpenAI SDK / curl
        |
        |  POST /v1/chat/completions
        v
+----------------------- TideGate -----------------------+
|  FastAPI edge                                           |
|  - request id, auth, OpenAI-compatible errors, SSE       |
|                         |                               |
|  Quota admission        |  Redis Lua token buckets       |
|  - RPM / TPM / streams / monthly budget                 |
|                         |                               |
|  Cache                  |  L1 exact -> L2 semantic       |
|  - replay cached answers as SSE when the client streams |
|                         |                               |
|  Router                 |  P2C, local breakers, fallback |
|  - hedge slow streams, cascade cheap drafts when useful |
|                         |                               |
|  Provider adapters      |  httpx streaming clients       |
+-------------+----------------------------+--------------+
              |                            |
              v                            v
        Redis Stack                  PostgreSQL
        quota/cache/routing state     usage ledger
```

## What It Does

**Streaming proxy.** The public API looks like OpenAI's chat completions endpoint, including SSE. The gateway keeps upstream streams inside cancellable `httpx` contexts, so a client disconnect closes the provider stream instead of letting token generation run in the background.

**Token quota.** Each tenant gets Redis-backed admission control for request rate, token rate, concurrent streams, and monthly budget. The gateway reserves from an estimate before dispatch, then settles with actual usage when the response finishes.

**Two-level cache.** L1 is an exact Redis cache over the normalized request. L2 uses embeddings for recall and a cross-encoder reranker for the final semantic-cache decision. Cache hits can still be returned as paced SSE chunks, so clients do not need a separate non-streaming path.

**Routing and breakers.** A logical model can point at several upstream deployments. Routing uses local EWMA stats with power-of-two choices, filters open breakers, and can fall back to a smaller model group or stale cache when configured.

**Tail-latency control.** Streaming hedges start a second upstream attempt when the primary is slow enough and the hedge budget allows it. The loser is cancelled, and the benchmark report keeps the before/after P99 numbers visible.

## Why Python Here

Go would be a reasonable choice for a gateway, but the hard part in this project is not raw socket throughput. Most of the work is I/O-bound streaming, timeout boundaries, Redis/Postgres correctness, cancellation, and policy decisions around cache and routing. Python 3.12 with `asyncio`, FastAPI, `httpx`, and process pools is enough for that shape of service, while keeping the model-side experiments and calibration scripts in the same language.

The boundary is explicit: CPU-heavy token counting and embedding/reranking work is pushed into `ProcessPoolExecutor`; the event loop should only coordinate I/O and small bookkeeping.

## Quick Start

Start Redis/Postgres:

```bash
make up
```

Run the local mock provider and gateway:

```bash
uv run --extra dev --extra test python -m mock_provider --host 127.0.0.1 --port 9001

TIDEGATE_ADMIN_TOKEN=dev-admin \
MOCK_A_KEY=mock-key \
MOCK_B_KEY=mock-key \
TIDEGATE_PG_DSN=postgresql://tidegate:tidegate@127.0.0.1:5432/tidegate \
uv run --extra dev --extra test python -m tidegate --config config/gateway.yaml
```

Send a streaming request with the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(
    api_key="<demo key>",
    base_url="http://127.0.0.1:8000/v1",
)

stream = client.chat.completions.create(
    model="chat-large",
    messages=[{"role": "user", "content": "Give me a short gateway smoke test."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

The demo key is intentionally local-only and matches the hash in `config/gateway.yaml`.

## Benchmark Snapshot

The numbers below come from `out/benchmark.md` in this checkout.

| Scenario | Result |
|---|---:|
| Gateway TTFT P99 | 94.598 ms |
| Gateway E2E P99 | 200.273 ms |
| Gateway overhead P99 | 4.950 ms |
| Peak streaming inflight | 3082 |
| Concurrency success rate | 0.982 |
| Loop lag peak during concurrency run | 0.002 s |
| Cache-hit TTFT P50 | 6.123 ms |
| L1 hit rate in cache-hit run | 0.412 |
| Hedge TTFT P99, off -> on | 1773.046 ms -> 295.464 ms |
| Hedge P99 reduction | 83.3% |

## Verification

```bash
make check
make test
make up && uv run --extra dev --extra test pytest -m integration tests/integration && make down
```

## Status / Scope

This is a personal engineering project, not a production service with real online traffic. The benchmark uses a deterministic mock provider so gateway behavior can be reproduced without depending on a paid model API or a flaky external provider.

The project deliberately does not try to be an agent framework, a RAG stack, or a frontend product. The useful surface area is LLM traffic governance: compatible streaming, quota, cache, provider routing, failure handling, tail latency, and usage accounting.
