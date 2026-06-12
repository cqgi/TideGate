from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn

from tidegate.app import create_app
from tidegate.config.loader import load_config


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/gateway.yaml")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    uvicorn.run(create_app(config), host=config.server.host, port=config.server.port)


if __name__ == "__main__":
    main()
