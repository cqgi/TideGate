from __future__ import annotations

from pathlib import Path

from tidegate.config.loader import load_config


def test_load_config() -> None:
    """SPEC-M0-2."""
    config = load_config(Path("config/gateway.yaml"))
    assert config.server.port == 8000
    assert "chat-large" in config.model_groups
