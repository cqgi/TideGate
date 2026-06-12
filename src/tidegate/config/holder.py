from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from tidegate.config.loader import load_config
from tidegate.config.models import GatewayConfig


@dataclass(frozen=True)
class ReloadResult:
    ok: bool
    version: int
    error: str | None = None


class ConfigHolder:
    def __init__(self, initial: GatewayConfig, path: Path, version: int = 0) -> None:
        self._current = initial
        self._path = path
        self._version = version

    @property
    def current(self) -> GatewayConfig:
        return self._current

    @property
    def version(self) -> int:
        return self._version

    @property
    def path(self) -> Path:
        return self._path

    def reload(self, version: int | None = None) -> ReloadResult:
        try:
            next_config = load_config(self._path)
        except (OSError, ValueError, ValidationError) as exc:
            return ReloadResult(ok=False, version=self._version, error=str(exc))
        self._current = next_config
        if version is not None:
            self._version = version
        return ReloadResult(ok=True, version=self._version)

    def replace(self, next_config: GatewayConfig, *, version: int | None = None) -> None:
        self._current = next_config
        if version is not None:
            self._version = version
