# TideGate

TideGate 是一个 OpenAI 协议兼容的 LLM 推理网关。我做这个项目是为了解决多个业务共享同一个模型 API 时的痛点：直接让每个服务连接供应商会导致成本不透明、故障难以隔离、重复请求浪费钱、流式响应容易出 bug。这个项目的目标很清晰——在客户端和供应商池之间放一个网关，把配额、缓存、路由和尾延迟的行为都显式化地管理起来。

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

## 核心能力

**流式代理。** 公开 API 兼容 OpenAI 聊天补全端点，包括 SSE 流式。网关在可取消的 `httpx` 上下文中保持上游连接，这样客户端断连时会立即关闭上游生成，而不是让它在后台继续烧 token。

**Token 级限流。** 每个租户都有 Redis 背书的准入控制，维度包括请求速率、Token 速率、并发流数、月度预算。网关根据估算值预先扣减额度，请求结束时拿到实际用量后再结算差额。

**两级缓存。** L1 是精确匹配的 Redis 缓存，键是规范化后的请求 SHA256。L2 用 embedding 做召回、cross-encoder 重排来做最终的语义相似度判断。命中缓存也能回放为 SSE 分段流式，客户端不需要单独的非流式路径。

**路由和熔断。** 一个逻辑模型可以指向多个上游部署。路由使用本地 EWMA 统计 + 两随机选择，会过滤掉打开状态的熔断器，支持级联到更小的模型或陈旧缓存。

**尾延迟治理。** 当主请求迟迟不出首 token 时，网关会发起对冲请求到另一个上游。谁先出 token 就用谁，对手立即取消。基准测试报告里能看到前后的 P99 对比。

## 为什么用 Python

Go 对网关来说是合理的选择，但这个项目的难点不在原始的网络吞吐。真正的工作是 I/O 密集的流式处理、超时边界、Redis/Postgres 的正确性、取消语义、以及缓存和路由的策略决策。Python 3.12 的 `asyncio`、FastAPI、`httpx` 和进程池足以胜任这个工作量，同时能把模型实验和标定脚本都用同一种语言写。

边界很清晰：CPU 密集的工作（token 计数、embedding、重排）都推进 `ProcessPoolExecutor`；事件循环只负责 I/O 协调和小的簿记。

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

下表来自仓库里的 `out/benchmark.md`。

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

## 项目状态和范围

这是一个个人工程项目，没有真实的线上流量。基准测试使用确定性的 mock 供应商，这样网关行为可以在不依赖付费 API 或不稳定外部供应商的情况下复现。

项目有意不尝试成为 Agent 框架、RAG 栈或前端产品。关注的是 LLM 流量治理这一个方向：兼容流式、配额、缓存、供应商路由、故障处理、尾延迟优化、使用量计费。
