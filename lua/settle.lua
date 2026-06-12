local tpm_key = KEYS[1]
local conc_key = KEYS[2]
local budget_key = KEYS[3]
local resv_zset = KEYS[4]
local resv_hash = KEYS[5]

local request_id = ARGV[1]
local actual_tpm = tonumber(ARGV[2])
local actual_budget_micro = tonumber(ARGV[3])

local raw = redis.call("HGET", resv_hash, request_id)
if raw == false then
  return {0}
end

local data = cjson.decode(raw)
local tpm_est = tonumber(data["tpm_est"])
local budget_est_micro = tonumber(data["budget_est_micro"])
local tpm_cap = tonumber(redis.call("HGET", tpm_key, "cap") or "0")
local tokens = tonumber(redis.call("HGET", tpm_key, "tokens") or "0")
local tpm_next = tokens + (tpm_est - actual_tpm)
if tpm_cap > 0 then
  tpm_next = math.min(tpm_cap, tpm_next)
end

redis.call("HSET", tpm_key, "tokens", tpm_next)
redis.call("INCRBY", budget_key, budget_est_micro - actual_budget_micro)
local conc = tonumber(redis.call("GET", conc_key) or "0")
redis.call("SET", conc_key, math.max(0, conc - 1))
redis.call("ZREM", resv_zset, request_id)
redis.call("HDEL", resv_hash, request_id)

return {1}
