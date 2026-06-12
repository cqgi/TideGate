from __future__ import annotations

from dataclasses import dataclass

from tidegate.config.models import CascadeConfig, DeploymentConfig, GatewayConfig, PolicyConfig
from tidegate.core.models import UnifiedRequest, UnifiedResponse


@dataclass(frozen=True)
class CascadeDecision:
    enabled: bool
    draft_group: str | None = None
    threshold: float = 0.0
    reason: str | None = None


def cascade_decision(
    req: UnifiedRequest,
    policy: PolicyConfig,
    settings: GatewayConfig,
) -> CascadeDecision:
    cascade = policy.cascade
    if not cascade.enabled:
        return CascadeDecision(False, reason="disabled")
    if req.stream:
        return CascadeDecision(False, reason="stream")
    if req.has_tools:
        return CascadeDecision(False, reason="tools")
    if cascade.draft_model_group is None or cascade.draft_model_group not in settings.model_groups:
        return CascadeDecision(False, reason="missing_draft_group")
    if not any(
        deployment.supports_logprobs
        for deployment in settings.model_groups[cascade.draft_model_group].deployments
    ):
        return CascadeDecision(False, reason="draft_no_logprobs")
    return CascadeDecision(
        True,
        draft_group=cascade.draft_model_group,
        threshold=cascade.threshold,
    )


def force_logprobs(req: UnifiedRequest) -> UnifiedRequest:
    raw_body = dict(req.raw_body)
    raw_body["logprobs"] = True
    return req.model_copy(update={"logprobs": True, "raw_body": raw_body})


def draft_accepted(response: UnifiedResponse, cascade: CascadeConfig) -> bool:
    if cascade.confidence_metric != "mean_logprob":
        return False
    return response.mean_logprob is not None and response.mean_logprob >= cascade.threshold


def deployment_supports_logprobs(deployment: DeploymentConfig) -> bool:
    return deployment.supports_logprobs
