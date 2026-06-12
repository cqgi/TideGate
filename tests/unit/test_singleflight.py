from __future__ import annotations

import pytest

from tidegate.cache.singleflight import SingleFlight


@pytest.mark.asyncio
async def test_leader_success_clears_future_and_wakes_follower() -> None:
    """SPEC-M4-3."""
    sf: SingleFlight[str] = SingleFlight()
    leader = sf.acquire("k")
    follower = sf.acquire("k")

    assert leader.leader
    assert not follower.leader
    sf.resolve(leader, "ok")
    sf.release(leader)

    assert await follower.future == "ok"
    assert sf.pending_count() == 0


@pytest.mark.asyncio
async def test_leader_failure_clears_future_and_follower_can_fallback() -> None:
    """SPEC-M4-3."""
    sf: SingleFlight[str] = SingleFlight()
    leader = sf.acquire("k")
    follower = sf.acquire("k")

    try:
        sf.reject(leader, RuntimeError("boom"))
    finally:
        sf.release(leader)

    with pytest.raises(RuntimeError):
        await follower.future
    retry = sf.acquire("k")
    assert retry.leader
    assert sf.pending_count() == 1
    sf.resolve(retry, "fallback-ok")
    sf.release(retry)
    assert await retry.future == "fallback-ok"
