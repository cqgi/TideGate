from __future__ import annotations

from dataclasses import dataclass

from tidegate.config.models import GatewayConfig, ModelGroupConfig, PolicyConfig, TenantConfig


@dataclass(frozen=True)
class RouteLevel:
    model_group_name: str
    group: ModelGroupConfig
    degraded: str | None = None


class RoutingLadder:
    def __init__(self, settings: GatewayConfig) -> None:
        self._settings = settings

    def levels(self, requested_model: str, tenant: TenantConfig) -> list[RouteLevel]:
        policy = self._settings.policies.get(tenant.policy, PolicyConfig())
        levels: list[RouteLevel] = []
        seen: set[str] = set()
        self._append(levels, seen, requested_model, degraded=None)

        chain = list(policy.fallback_chain)
        fallback_names = (
            chain[chain.index(requested_model) + 1 :] if requested_model in chain else chain
        )
        for name in fallback_names:
            self._append(levels, seen, name, degraded="fallback")

        smaller = policy.degradation.smaller_model_group
        if smaller is not None:
            self._append(levels, seen, smaller, degraded="smaller-model")
        return levels

    def stale_cache(self) -> None:
        # TODO(SPEC-M3-4): M4 wires stale-cache lookup into this hook.
        return None

    def _append(
        self,
        levels: list[RouteLevel],
        seen: set[str],
        name: str,
        *,
        degraded: str | None,
    ) -> None:
        if name in seen:
            return
        group = self._settings.model_groups.get(name)
        if group is None:
            return
        seen.add(name)
        levels.append(RouteLevel(name, group, degraded))
