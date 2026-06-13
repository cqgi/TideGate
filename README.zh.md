# TideGate

TideGate 是一个 OpenAI 协议兼容的 LLM 推理网关，位于 SDK 客户端和供应商池之间，负责配额准入、流式代理、缓存查询、供应商选择、对冲请求和用量结算。

```
OpenAI SDK / curl
        |
        |  POST /v1/chat/completions
        v
+----------------------- TideGate -----------------------+
|  FastAPI 边缘层                                        |
|  - 请求 ID、鉴权、OpenAI 兼容错误响应、SSE 流式      |
|                         |                               |
|  配额准入               |  Redis Lua 令牌桶            |
|  - RPM / TPM / 并发流数 / 月度预算                    |
|                         |                               |
|  缓存                   |  L1 精确 -> L2 语义          |
|  - 命中时按 SSE 分段回放缓存答案                       |
|                         |                               |
|  路由                   |  P2C、本地熔断、降级         |
|  - 对冲慢请求、廉价小模型级联                          |
|                         |                               |
|  供应商适配器           |  httpx 流式客户端            |
+-------------+----------------------------+--------------+
              |                            |
              v                            v
        Redis Stack                  PostgreSQL
        配额/缓存/路由状态             使用量账本
```

## 核心组件

**流式 API。** `POST /v1/chat/completions` 兼容 OpenAI 聊天补全接口形态，包括 SSE 响应。上游流运行在可取消的 `httpx` 上下文中，客户端断连时会关闭供应商流。

**配额准入。** 每个租户都有 Redis 背书的请求速率、Token 速率、并发流数和月度预算限制。Lua 脚本在转发前原子化完成检查和预留；请求结束后根据实际用量结算差额。

**缓存。** L1 是基于规范化请求的 Redis 精确缓存。L2 用 embedding 召回加 cross-encoder 重排做语义缓存判断。缓存命中也可以按 SSE 分段回放。

**路由。** 一个逻辑模型可以指向多个上游部署。选择逻辑基于本地 EWMA 统计和两随机选择，过滤打开状态的熔断器，并在配置允许时降级到更小模型组或陈旧缓存。

**尾延迟。** 当主请求在首 token 前变慢且对冲预算允许时，网关可以向另一个上游发起第二次尝试。任一流胜出后，另一个尝试会被取消。

## 运行模型

网关路径使用 FastAPI、`httpx` 和 `asyncio`。Token 计数、embedding、重排等 CPU 密集工作运行在 `ProcessPoolExecutor` 中，事件循环只负责 I/O 协调和少量簿记。

## 快速开始

启动 Redis 和 Postgres：

```bash
make up
```

启动本地 mock 供应商和网关：

```bash
uv run --extra dev --extra test python -m mock_provider --host 127.0.0.1 --port 9001

TIDEGATE_ADMIN_TOKEN=dev-admin \
MOCK_A_KEY=mock-key \
MOCK_B_KEY=mock-key \
TIDEGATE_PG_DSN=postgresql://tidegate:tidegate@127.0.0.1:5432/tidegate \
uv run --extra dev --extra test python -m tidegate --config config/gateway.yaml
```

用 OpenAI SDK 发送流式请求：

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

Demo 密钥是本地专用的，和 `config/gateway.yaml` 里的哈希值一致。

## 基准测试数据

下表来自 `out/benchmark.md`。

| 场景 | 结果 |
|---|---:|
| 网关 TTFT P99 | 94.598 ms |
| 网关 E2E P99 | 200.273 ms |
| 网关开销 P99 | 4.950 ms |
| 流式并发峰值 | 3082 |
| 并发成功率 | 0.982 |
| 并发运行中事件循环延迟峰值 | 0.002 s |
| 缓存命中 TTFT P50 | 6.123 ms |
| 缓存命中运行中 L1 命中率 | 0.412 |
| 对冲 TTFT P99（关闭 -> 开启） | 1773.046 ms -> 295.464 ms |
| 对冲 P99 降幅 | 83.3% |

## 验证

```bash
make check
make test
make up && uv run --extra dev --extra test pytest -m integration tests/integration && make down
```

## 说明

基准测试使用确定性的 mock 供应商，便于在不依赖外部模型 API 的情况下复现延迟、故障转移和缓存行为。

不包含：Agent 工作流、前端 UI、完整 RAG 应用逻辑。
