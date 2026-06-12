from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FrozenModel(BaseModel):
    # REWORK-M0-5: gateway config typos must fail validation instead of being ignored.
    model_config = ConfigDict(frozen=True, extra="forbid")


class ServerConfig(FrozenModel):
    host: str = Field(default="0.0.0.0", description="Bind host")
    port: int = Field(default=8000, description="Bind port")
    sse_heartbeat_interval_s: float = Field(default=15.0, description="SSE idle heartbeat")
    auth_cache_size: int = Field(default=1024, description="In-process API key cache capacity")
    auth_cache_ttl_s: float = Field(default=60.0, description="In-process API key cache TTL")
    loop_lag_interval_s: float = Field(default=1.0, description="Loop lag probe interval")
    config_poll_interval_s: float = Field(default=30.0, description="Config version poll interval")
    provider_pool_drain_s: float = Field(default=60.0, description="Old provider pool drain delay")
    config_reload_backoff_initial_s: float = Field(
        default=1.0, description="Initial Redis hot-reload reconnect backoff"
    )
    config_reload_backoff_max_s: float = Field(
        default=60.0, description="Maximum Redis hot-reload reconnect backoff"
    )
    cpu_pool_workers: int = Field(default=2, description="Shared CPU process pool workers")


class TimeoutConfig(FrozenModel):
    connect_s: float = Field(default=2.0, description="Upstream connection timeout")
    ttft_s: float = Field(default=10.0, description="Time to first token budget")
    inter_chunk_s: float = Field(default=15.0, description="Upstream read timeout")
    total_s: float = Field(default=300.0, description="Total gateway request budget")


class RedisConfig(FrozenModel):
    url: str = Field(default="redis://localhost:6379/0", description="Redis Stack URL")


class PostgresConfig(FrozenModel):
    dsn_env: str = Field(default="TIDEGATE_PG_DSN", description="PostgreSQL DSN env name")


class ProviderConfig(FrozenModel):
    type: str
    base_url: str
    api_key_env: str
    max_connections: int = Field(default=200, description="HTTP max connections")


class DeploymentConfig(FrozenModel):
    provider: str
    upstream_model: str
    weight: int = Field(default=1, description="Routing weight")
    price_per_1k_input_usd: float = Field(default=0.0, description="Input token price")
    price_per_1k_output_usd: float = Field(default=0.0, description="Output token price")
    supports_logprobs: bool = Field(
        default=False, description="Whether deployment supports logprobs"
    )


class ModelGroupConfig(FrozenModel):
    deployments: tuple[DeploymentConfig, ...]


class CacheToggleConfig(FrozenModel):
    l1: bool = True
    l2: bool = False


class TenantConfig(FrozenModel):
    id: str
    api_key_sha256: str
    plan: str = "free"
    policy: str = "default"
    cache: CacheToggleConfig = Field(default_factory=CacheToggleConfig)


class BreakerConfig(FrozenModel):
    window_size: int = 20
    failure_rate_to_open: float = 0.5
    min_samples: int = 10
    open_cooldown_s: float = 30.0
    cooldown_max_s: float = 300.0
    half_open_probes: int = 3


class RoutingConfig(FrozenModel):
    ewma_alpha: float = 0.2
    slow_call_ttft_slo_s: float = 8.0
    breaker: BreakerConfig = Field(default_factory=BreakerConfig)
    p2c_weights: dict[str, float] = Field(
        default_factory=lambda: {"ttft": 0.4, "error_rate": 0.3, "inflight": 0.2, "price": 0.1}
    )
    max_attempts_before_first_byte: int = 3
    agg_report_interval_s: float = 5.0
    agg_ttl_s: int = Field(default=60, description="Redis aggregation key TTL")


class DegradationConfig(FrozenModel):
    smaller_model_group: str | None = None
    stale_cache: bool = True


class HedgingConfig(FrozenModel):
    enabled: bool = False
    trigger_quantile: float = 0.95
    trigger_floor_s: float = 2.0
    max_hedge_ratio: float = 0.05


class CascadeConfig(FrozenModel):
    enabled: bool = False
    draft_model_group: str | None = None
    confidence_metric: str = "mean_logprob"
    threshold: float = -0.45


class PolicyConfig(FrozenModel):
    fallback_chain: tuple[str, ...] = ("chat-large",)
    degradation: DegradationConfig = Field(default_factory=DegradationConfig)
    hedging: HedgingConfig = Field(default_factory=HedgingConfig)
    cascade: CascadeConfig = Field(default_factory=CascadeConfig)


class QuotaPlanConfig(FrozenModel):
    rpm: int = 60
    tpm: int = 100000
    concurrent_streams: int = 10
    monthly_budget_usd: float = 10.0
    fail_mode: Literal["open", "closed"] = "closed"


class L1CacheConfig(FrozenModel):
    ttl_s: int = 86400
    ttl_jitter_ratio: float = 0.1
    max_value_bytes: int = 262144


class L2CacheConfig(FrozenModel):
    similarity_threshold: float = 0.92
    max_temperature: float = 0.3
    index_capacity: int = 100000
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embed_pool_workers: int = 2


class CacheConfig(FrozenModel):
    l1: L1CacheConfig = Field(default_factory=L1CacheConfig)
    l2: L2CacheConfig = Field(default_factory=L2CacheConfig)
    volatile_intent_patterns: tuple[str, ...] = ("今天", "现在", "最新", "几点", "股价", "天气")
    replay_chunk_chars: int = 24
    replay_interval_ms: int = 15


class QuotaEstimatorConfig(FrozenModel):
    output_p95_fallback: int = 1024
    correction_ewma_alpha: float = 0.1


class SettlementConfig(FrozenModel):
    batch_size: int = 100
    batch_interval_ms: int = 200
    queue_max: int = 10000


class SweeperConfig(FrozenModel):
    interval_s: int = 10
    batch_limit: int = 100
    reservation_ttl_s: int = 600
    settle_timeout_s: float = 1.0


class GatewayConfig(FrozenModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    timeouts: TimeoutConfig = Field(default_factory=TimeoutConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    providers: dict[str, ProviderConfig]
    model_groups: dict[str, ModelGroupConfig]
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    # REWORK-M0-5: reserve full contract schema sections even before later milestones use them.
    policies: dict[str, PolicyConfig] = Field(default_factory=dict)
    quota_plans: dict[str, QuotaPlanConfig] = Field(default_factory=dict)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    quota_estimator: QuotaEstimatorConfig = Field(default_factory=QuotaEstimatorConfig)
    settlement: SettlementConfig = Field(default_factory=SettlementConfig)
    sweeper: SweeperConfig = Field(default_factory=SweeperConfig)
    tenants: tuple[TenantConfig, ...]

    def tenant_by_id(self, tenant_id: str) -> TenantConfig | None:
        return next((tenant for tenant in self.tenants if tenant.id == tenant_id), None)
