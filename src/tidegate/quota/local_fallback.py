from __future__ import annotations

import os
import time
from dataclasses import dataclass

from tidegate.config.models import QuotaPlanConfig
from tidegate.quota.estimator import Estimate


@dataclass
class _Bucket:
    tokens: float
    ts: float


class LocalFallbackLimiter:
    def __init__(self) -> None:
        self._rpm: dict[str, _Bucket] = {}
        self._tpm: dict[str, _Bucket] = {}

    def allow(self, tenant_id: str, plan: QuotaPlanConfig, estimate: Estimate) -> bool:
        instances = max(1, int(os.environ.get("TIDEGATE_INSTANCE_COUNT", "1")))
        rpm_cap = max(1.0, plan.rpm / instances)
        tpm_cap = max(1.0, plan.tpm / instances)
        rpm_rate = rpm_cap / 60
        tpm_rate = tpm_cap / 60
        now = time.monotonic()
        rpm_bucket = self._bucket(self._rpm, tenant_id, rpm_cap, now)
        tpm_bucket = self._bucket(self._tpm, tenant_id, tpm_cap, now)
        rpm_tokens = self._refill(rpm_bucket, rpm_rate, rpm_cap, now)
        tpm_tokens = self._refill(tpm_bucket, tpm_rate, tpm_cap, now)
        if rpm_tokens < 1 or tpm_tokens < estimate.tpm_cost:
            return False
        rpm_bucket.tokens -= 1
        tpm_bucket.tokens -= estimate.tpm_cost
        return True

    def _bucket(
        self,
        buckets: dict[str, _Bucket],
        tenant_id: str,
        cap: float,
        now: float,
    ) -> _Bucket:
        bucket = buckets.get(tenant_id)
        if bucket is None:
            bucket = _Bucket(tokens=cap, ts=now)
            buckets[tenant_id] = bucket
        return bucket

    def _refill(self, bucket: _Bucket, rate: float, cap: float, now: float) -> float:
        bucket.tokens = min(cap, bucket.tokens + (now - bucket.ts) * rate)
        bucket.ts = now
        return bucket.tokens
