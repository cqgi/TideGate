from __future__ import annotations

from tidegate.config.models import (
    CascadeConfig,
    DeploymentConfig,
    GatewayConfig,
    ModelGroupConfig,
    PolicyConfig,
    ProviderConfig,
    TenantConfig,
)
from tidegate.core.models import ChatMessage, UnifiedRequest, UnifiedResponse, Usage
from tidegate.routing.cascade import cascade_decision, draft_accepted, force_logprobs


def test_cascade_bypasses_stream_and_tools() -> None:
    policy = PolicyConfig(cascade=CascadeConfig(enabled=True, draft_model_group="chat-small"))
    settings = _settings()

    assert cascade_decision(_request(stream=True), policy, settings).reason == "stream"
    assert cascade_decision(_request(has_tools=True), policy, settings).reason == "tools"


def test_cascade_enables_non_stream_draft_group_and_forces_logprobs() -> None:
    policy = PolicyConfig(cascade=CascadeConfig(enabled=True, draft_model_group="chat-small"))
    decision = cascade_decision(_request(), policy, _settings())
    draft_req = force_logprobs(_request())

    assert decision.enabled
    assert decision.draft_group == "chat-small"
    assert draft_req.logprobs is True
    assert draft_req.raw_body["logprobs"] is True


def test_draft_acceptance_uses_mean_logprob_threshold() -> None:
    cascade = CascadeConfig(enabled=True, threshold=-0.45)

    assert draft_accepted(_response(-0.3), cascade)
    assert not draft_accepted(_response(-0.8), cascade)
    assert not draft_accepted(_response(None), cascade)


def _request(*, stream: bool = False, has_tools: bool = False) -> UnifiedRequest:
    return UnifiedRequest(
        request_id="req",
        tenant_id="demo",
        model="chat-large",
        messages=[ChatMessage(role="user", content="hi")],
        stream=stream,
        has_tools=has_tools,
        raw_body={"model": "chat-large", "messages": [{"role": "user", "content": "hi"}]},
    )


def _response(mean_logprob: float | None) -> UnifiedResponse:
    return UnifiedResponse(
        content="ok",
        finish_reason="stop",
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model="mock",
        mean_logprob=mean_logprob,
    )


def _settings() -> GatewayConfig:
    return GatewayConfig(
        providers={
            "mock-a": ProviderConfig(type="openai_compatible", base_url="", api_key_env="X")
        },
        model_groups={
            "chat-large": ModelGroupConfig(
                deployments=(
                    DeploymentConfig(
                        provider="mock-a",
                        upstream_model="mock-gpt-large",
                        supports_logprobs=True,
                    ),
                )
            ),
            "chat-small": ModelGroupConfig(
                deployments=(
                    DeploymentConfig(
                        provider="mock-a",
                        upstream_model="mock-gpt-small",
                        supports_logprobs=True,
                    ),
                )
            ),
        },
        tenants=(TenantConfig(id="demo", api_key_sha256="hash"),),
    )
