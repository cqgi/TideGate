# TideGate Benchmark Report

| run | requests | success_rate | ttft_p50_ms | ttft_p95_ms | ttft_p99_ms | e2e_p99_ms | gateway_overhead_p99_ms | loop_lag_peak_s | peak_inflight | rss_mb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 202 | 1.000 | 89.805 | 98.583 | 106.551 | 198.836 | 4.950 | 0.001 | 8 | 131.750 |
| gateway | 423 | 1.000 | 87.872 | 92.042 | 94.598 | 200.273 | 4.950 | 0.000 | 11 | 131.797 |
| concurrency | 3090 | 0.982 | 94.944 | 19935.517 | 27610.264 | 61923.375 | 4.983 | 0.002 | 3082 | 335.188 |
| cache-hit | 391 | 1.000 | 87.298 | 94.611 | 100.805 | 190.676 | 4.950 | 0.001 | 7 | 132.000 |
| hedge-off | 199 | 1.000 | 91.178 | 534.077 | 1773.046 | 1799.820 | 4.950 | 0.001 | 8 | 135.125 |
| hedge-on | 202 | 1.000 | 96.314 | 239.672 | 295.464 | 324.806 | 4.950 | 0.001 | 8 | 134.812 |

## Scenario Details

### baseline

- Cache headers: `{"miss": 202}`
- L1 hit rate: 
- Cache hit TTFT P50/P95/P99:  /  /  ms
- Mock directive: `{}`

### gateway

- Cache headers: `{"miss": 423}`
- L1 hit rate: 
- Cache hit TTFT P50/P95/P99:  /  /  ms
- Mock directive: `{}`

### concurrency

- Cache headers: `{"bypass": 55, "miss": 2948}`
- L1 hit rate: 
- Cache hit TTFT P50/P95/P99:  /  /  ms
- Mock directive: `{"output_tokens": 100, "tpot_ms": 300}`

### cache-hit

- Cache headers: `{"hit-exact": 161, "miss": 230}`
- L1 hit rate: 0.412
- Cache hit TTFT P50/P95/P99: 6.123 / 10.378 / 17.671 ms
- Mock directive: `{}`

### hedge-off

- Cache headers: `{"miss": 199}`
- L1 hit rate: 
- Cache hit TTFT P50/P95/P99:  /  /  ms
- Mock directive: `{"output_tokens": 8, "tpot_ms": 2, "ttft_lognorm": "4.5,1.0"}`

### hedge-on

- Cache headers: `{"miss": 202}`
- L1 hit rate: 
- Cache hit TTFT P50/P95/P99:  /  /  ms
- Mock directive: `{"output_tokens": 8, "tpot_ms": 2, "ttft_lognorm": "4.5,1.0"}`

## Hedge Effect

- TTFT P99 off: 1773.046 ms
- TTFT P99 on: 295.464 ms
- P99 improvement: 1477.582 ms

## Report Notes

- TTFT absolute values reflect mock provider speed. Gateway capacity is read from baseline/gateway deltas, gateway_overhead_p99_ms, and loop_lag_peak_s.
- Concurrency target is validated by peak_inflight; loop_lag_peak_s is reported as observed instead of capped or smoothed.
