CREATE TABLE IF NOT EXISTS attempt_ledger (
  request_id TEXT NOT NULL,
  attempt_seq SMALLINT NOT NULL,
  tenant_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  outcome TEXT NOT NULL,
  provider TEXT,
  upstream_model TEXT,
  prompt_tokens INT,
  completion_tokens INT,
  usage_source TEXT NOT NULL,
  cost_tenant_microusd BIGINT NOT NULL DEFAULT 0,
  cost_platform_microusd BIGINT NOT NULL DEFAULT 0,
  bearer_reason TEXT,
  error_category TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (request_id, attempt_seq)
);

CREATE INDEX IF NOT EXISTS idx_attempt_ledger_tenant_created
ON attempt_ledger (tenant_id, created_at);

CREATE INDEX IF NOT EXISTS idx_attempt_ledger_platform
ON attempt_ledger (created_at) WHERE cost_platform_microusd > 0;
