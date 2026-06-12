from __future__ import annotations

import random

from tidegate.config.models import DeploymentConfig, ModelGroupConfig
from tidegate.core.errors import ErrorCategory, GatewayError


def pick(group: ModelGroupConfig, exclude: set[tuple[str, str]]) -> DeploymentConfig:
    candidates = [
        deployment
        for deployment in group.deployments
        if (deployment.provider, deployment.upstream_model) not in exclude and deployment.weight > 0
    ]
    if not candidates:
        raise GatewayError("no deployment available", ErrorCategory.RETRYABLE_UPSTREAM)
    total = sum(deployment.weight for deployment in candidates)
    # SPEC-M1-3: weighted random selector is a narrow M1 placeholder for M3 P2C.
    target = random.uniform(0, total)
    cursor = 0.0
    for deployment in candidates:
        cursor += deployment.weight
        if cursor >= target:
            return deployment
    return candidates[-1]
