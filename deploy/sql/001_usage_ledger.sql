CREATE TABLE IF NOT EXISTS usage_ledger (
  request_id TEXT PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  model TEXT,
  provider TEXT,
  upstream_model TEXT,
  prompt_tokens INT,
  completion_tokens INT,
  cost_microusd BIGINT,
  cache_status TEXT,
  route_path TEXT,
  degraded TEXT,
  outcome TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_usage_ledger_tenant_created
ON usage_ledger (tenant_id, created_at);
