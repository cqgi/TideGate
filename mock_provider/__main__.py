from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn

from mock_provider.app import create_app
from mock_provider.generator import MockDefaults


def parse_lognorm(raw: str | None) -> tuple[float, float] | None:
    if raw is None:
        return None
    mu, sigma = raw.split(",", maxsplit=1)
    return float(mu), float(sigma)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--ttft-ms", type=int, default=80)
    parser.add_argument("--tpot-ms", type=int, default=5)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--ttft-lognorm")
    args = parser.parse_args(argv)

    lognorm = parse_lognorm(args.ttft_lognorm)
    defaults = MockDefaults(
        ttft_ms=args.ttft_ms,
        tpot_ms=args.tpot_ms,
        output_tokens=args.output_tokens,
        ttft_lognorm_mu=None if lognorm is None else lognorm[0],
        ttft_lognorm_sigma=None if lognorm is None else lognorm[1],
    )
    uvicorn.run(create_app(defaults), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
