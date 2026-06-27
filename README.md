# TideGate

TideGate is an OpenAI-compatible gateway for routing and accounting LLM chat-completion traffic. It sits between SDK clients and a small provider pool, then handles quota admission, streaming proxying, cache lookup, provider selection, hedging, and usage settlement.

![TideGate architecture](assets/tidegate-architecture.png)

## Core Components

**Streaming API.** `POST /v1/chat/completions` follows the OpenAI chat-completions shape, including SSE responses. Upstream streams run inside cancellable `httpx` contexts, so a client disconnect closes the provider stream.

**Quota admission.** Each tenant has Redis-backed limits for request rate, token rate, concurrent streams, and monthly budget. Lua scripts check and reserve quota atomically before dispatch; settlement adjusts the reservation after actual usage is known.

**Cache.** L1 is an exact Redis cache over the normalized request. L2 uses embedding recall plus a cross-encoder reranker for semantic-cache decisions. Cache hits can be replayed as SSE chunks.

**Routing.** A logical model can point at several upstream deployments. Selection uses power-of-two choices over local EWMA stats, filters open breakers, and can fall back to a smaller model group or stale cache when configured.

**Tail latency.** Hedging can start a second upstream attempt before the first token arrives when the primary is slow and the hedge budget allows it. After one stream wins, the other attempt is cancelled.

## Runtime Notes

FastAPI, `httpx`, and `asyncio` handle the gateway path. Token counting and embedding/reranking work run in `ProcessPoolExecutor` so the event loop stays on I/O coordination and small bookkeeping.

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

## Benchmarks

The numbers below are from `out/benchmark.md`.

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

## Notes

The benchmark uses a deterministic mock provider so latency, failover, and cache behavior can be reproduced without external model APIs.

Out of scope: agent workflows, frontend UI, and full RAG application logic.
