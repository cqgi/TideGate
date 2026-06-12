from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tidegate.config.models import GatewayConfig


def load_config(path: str | Path) -> GatewayConfig:
    raw = Path(path).read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw)
    if not isinstance(parsed, dict):
        raise ValueError("gateway config must be a YAML mapping")
    return GatewayConfig.model_validate(parsed)


def load_config_dict(data: dict[str, Any]) -> GatewayConfig:
    return GatewayConfig.model_validate(data)
