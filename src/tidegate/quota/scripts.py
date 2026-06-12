from __future__ import annotations

from pathlib import Path

import redis.asyncio as redis
from redis.exceptions import NoScriptError


class QuotaScripts:
    def __init__(self, redis_client: redis.Redis, script_dir: Path | None = None) -> None:
        self._redis = redis_client
        root = Path(__file__).resolve().parents[3]
        self._script_dir = script_dir or root / "lua"
        self._sources: dict[str, str] = {}
        self._shas: dict[str, str] = {}

    async def load(self) -> None:
        for name in ("check_and_reserve", "settle", "sweep"):
            source = (self._script_dir / f"{name}.lua").read_text(encoding="utf-8")
            self._sources[name] = source
            self._shas[name] = await self._redis.script_load(source)

    async def check_and_reserve(self, keys: list[str], args: list[str]) -> list[object]:
        return await self._eval("check_and_reserve", keys, args)

    async def settle(self, keys: list[str], args: list[str]) -> list[object]:
        return await self._eval("settle", keys, args)

    async def sweep(self, keys: list[str], args: list[str]) -> list[object]:
        return await self._eval("sweep", keys, args)

    async def _eval(self, name: str, keys: list[str], args: list[str]) -> list[object]:
        try:
            result = await self._redis.evalsha(self._shas[name], len(keys), *keys, *args)
        except NoScriptError:
            source = self._sources[name]
            self._shas[name] = await self._redis.script_load(source)
            result = await self._redis.evalsha(self._shas[name], len(keys), *keys, *args)
        if not isinstance(result, list):
            raise TypeError(f"unexpected lua result for {name}")
        return result
