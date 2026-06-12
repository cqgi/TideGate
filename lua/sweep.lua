local tpm_key = KEYS[1]
local conc_key = KEYS[2]
local budget_key = KEYS[3]
local resv_zset = KEYS[4]
local resv_hash = KEYS[5]

local now_ms = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local current_month = ARGV[3]

local request_ids = redis.call("ZRANGEBYSCORE", resv_zset, "-inf", now_ms, "LIMIT", 0, limit)
local count = 0
local tpm_cap = tonumber(redis.call("HGET", tpm_key, "cap") or "0")

for _, request_id in ipairs(request_ids) do
  local raw = redis.call("HGET", resv_hash, request_id)
  if raw ~= false then
    local data = cjson.decode(raw)
    local tpm_est = tonumber(data["tpm_est"])
    local budget_est_micro = tonumber(data["budget_est_micro"])
    local reservation_month = data["month"] or current_month
    local refund_budget_key = budget_key
    if reservation_month ~= current_month then
      -- DECISION: TideGate M2 targets standalone Redis, not cluster; dynamic same-prefix budget
      -- keys keep cross-month sweep refunds correct without changing Python key builders.
      local prefix = string.match(budget_key, "^(.*:budget:)")
      refund_budget_key = prefix .. reservation_month
    end
    local tokens = tonumber(redis.call("HGET", tpm_key, "tokens") or "0")
    local tpm_next = tokens + tpm_est
    if tpm_cap > 0 then
      tpm_next = math.min(tpm_cap, tpm_next)
    end
    redis.call("HSET", tpm_key, "tokens", tpm_next)
    redis.call("INCRBY", refund_budget_key, budget_est_micro)
    local conc = tonumber(redis.call("GET", conc_key) or "0")
    redis.call("SET", conc_key, math.max(0, conc - 1))
    count = count + 1
  end
  redis.call("ZREM", resv_zset, request_id)
  redis.call("HDEL", resv_hash, request_id)
end

return {count}
