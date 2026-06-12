from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Flight[T]:
    key: str
    future: asyncio.Future[T]
    leader: bool


class SingleFlight[T]:
    def __init__(self) -> None:
        self._futures: dict[str, asyncio.Future[T]] = {}

    def acquire(self, key: str) -> Flight[T]:
        future = self._futures.get(key)
        if future is not None:
            return Flight(key, future, False)
        future = asyncio.get_running_loop().create_future()
        self._futures[key] = future
        return Flight(key, future, True)

    def resolve(self, flight: Flight[T], value: T) -> None:
        if not flight.future.done():
            flight.future.set_result(value)

    def reject(self, flight: Flight[T], exc: BaseException) -> None:
        if not flight.future.done():
            flight.future.set_exception(exc)

    def release(self, flight: Flight[T]) -> None:
        self._futures.pop(flight.key, None)

    def pending_count(self) -> int:
        return len(self._futures)
