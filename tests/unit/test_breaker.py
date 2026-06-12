from __future__ import annotations

import pytest

from tidegate.config.models import BreakerConfig
from tidegate.routing.breaker import BreakerState, CircuitBreaker


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        ([False, False, False], BreakerState.CLOSED),
        ([False, True, True, True], BreakerState.OPEN),
        ([False, True, True, True, "advance"], BreakerState.HALF_OPEN),
        ([False, True, True, True, "advance", False, False], BreakerState.CLOSED),
        ([False, True, True, True, "advance", True], BreakerState.OPEN),
    ],
)
def test_breaker_transition_table(events: list[bool | str], expected: BreakerState) -> None:
    """SPEC-M3-2."""
    breaker = CircuitBreaker(
        BreakerConfig(
            window_size=4,
            failure_rate_to_open=0.5,
            min_samples=4,
            open_cooldown_s=1,
            cooldown_max_s=4,
            half_open_probes=2,
        )
    )
    now = 0.0

    for event in events:
        if event == "advance":
            now = breaker.open_until_s
            assert breaker.allow(now)
            continue
        breaker.record(success=not event, now_s=now)

    assert breaker.state == expected


def test_half_open_probe_limit() -> None:
    """SPEC-M3-2."""
    breaker = CircuitBreaker(
        BreakerConfig(
            window_size=2,
            failure_rate_to_open=0.5,
            min_samples=2,
            open_cooldown_s=1,
            cooldown_max_s=4,
            half_open_probes=1,
        )
    )
    breaker.record(success=False, now_s=0.0)
    breaker.record(success=True, now_s=0.0)

    assert breaker.state == BreakerState.OPEN
    assert breaker.allow(breaker.open_until_s)
    assert not breaker.allow(breaker.open_until_s)
