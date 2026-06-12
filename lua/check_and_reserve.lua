local rpm_key = KEYS[1]
local tpm_key = KEYS[2]
local conc_key = KEYS[3]
local budget_key = KEYS[4]
local resv_zset = KEYS[5]
local resv_hash = KEYS[6]

local now_ms = tonumber(ARGV[1])
local request_id = ARGV[2]
local resv_deadline_ms = tonumber(ARGV[3])
local rpm_rate = tonumber(ARGV[4])
local rpm_cap = tonumber(ARGV[5])
local tpm_rate = tonumber(ARGV[6])
local tpm_cap = tonumber(ARGV[7])
local tpm_cost = tonumber(ARGV[8])
local conc_cap = tonumber(ARGV[9])
local budget_cost_micro = tonumber(ARGV[10])
local budget_init_micro = tonumber(ARGV[11])

local function refill_bucket(key, rate, cap)
  local tokens = tonumber(redis.call("HGET", key, "tokens"))
  local ts_ms = tonumber(redis.call("HGET", key, "ts_ms"))
  if tokens == nil or ts_ms == nil then
    tokens = cap
    ts_ms = now_ms
  else
    tokens = math.min(cap, tokens + ((now_ms - ts_ms) * rate / 1000))
    ts_ms = now_ms
  end
  redis.call("HSET", key, "tokens", tokens, "ts_ms", ts_ms, "cap", cap)
  return tokens
end

local function retry_after_ms(shortage, rate)
  if rate <= 0 then
    return 1000
  end
  return math.ceil(shortage / rate * 1000)
end

local rpm_tokens = refill_bucket(rpm_key, rpm_rate, rpm_cap)
local tpm_tokens = refill_bucket(tpm_key, tpm_rate, tpm_cap)

redis.call("SETNX", budget_key, budget_init_micro)
redis.call("EXPIRE", budget_key, 3456000)
local conc = tonumber(redis.call("GET", conc_key) or "0")
local budget = tonumber(redis.call("GET", budget_key) or "0")

if rpm_tokens < 1 then
  return {0, "rpm", retry_after_ms(1 - rpm_tokens, rpm_rate)}
end
if tpm_tokens < tpm_cost then
  return {0, "tpm", retry_after_ms(tpm_cost - tpm_tokens, tpm_rate)}
end
if conc >= conc_cap then
  return {0, "concurrency", 1000}
end
if budget < budget_cost_micro then
  return {0, "budget", 1000}
end

redis.call("HINCRBYFLOAT", rpm_key, "tokens", -1)
redis.call("HINCRBYFLOAT", tpm_key, "tokens", -tpm_cost)
redis.call("INCR", conc_key)
redis.call("DECRBY", budget_key, budget_cost_micro)
redis.call("ZADD", resv_zset, resv_deadline_ms, request_id)
redis.call(
  "HSET",
  resv_hash,
  request_id,
  cjson.encode({tpm_est = tpm_cost, budget_est_micro = budget_cost_micro})
)

return {1}
