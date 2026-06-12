from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from tidegate.cache.embedding import embed_sync, init_embedding_worker


@dataclass(frozen=True)
class Pair:
    a: str
    b: str
    label: int


@dataclass(frozen=True)
class ScanRow:
    tau: float
    hit_rate: float
    false_hit_rate: float
    hits: int
    false_hits: int


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/calibration/semcache_pairs.jsonl")
    parser.add_argument("--max-false-hit", type=float, default=0.01)
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--model-cache-dir", default=".cache/huggingface")
    parser.add_argument("--hf-endpoint", default="")
    parser.add_argument("--output", default="config/calibrated.yaml")
    args = parser.parse_args(argv)

    pairs_path = Path(args.pairs)
    raw = pairs_path.read_bytes()
    pairs = _load_pairs(raw)
    dataset_sha256 = hashlib.sha256(raw).hexdigest()

    init_embedding_worker(args.model, args.model_cache_dir, args.hf_endpoint)
    vectors = _embed_texts(pairs)
    similarities = [(float(np.dot(vectors[pair.a], vectors[pair.b])), pair.label) for pair in pairs]
    rows = _scan(similarities)
    selected = _select(rows, max_false_hit=args.max_false_hit)
    operating_points = _operating_points(rows)
    positives = sum(1 for pair in pairs if pair.label == 1)
    negatives = len(pairs) - positives

    _print_report(
        pairs_path=pairs_path,
        model=args.model,
        dataset_sha256=dataset_sha256,
        pairs=len(pairs),
        positives=positives,
        negatives=negatives,
        rows=rows,
        selected=selected,
        operating_points=operating_points,
    )
    _write_calibrated(
        Path(args.output),
        model=args.model,
        pairs_path=pairs_path,
        dataset_sha256=dataset_sha256,
        pair_count=len(pairs),
        positives=positives,
        negatives=negatives,
        max_false_hit=args.max_false_hit,
        selected=selected,
        operating_points=operating_points,
    )


def _load_pairs(raw: bytes) -> list[Pair]:
    pairs: list[Pair] = []
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid jsonl at line {line_no}: {exc}") from exc
        a = payload.get("a")
        b = payload.get("b")
        label = payload.get("label")
        if not isinstance(a, str) or not isinstance(b, str) or label not in {0, 1}:
            raise SystemExit(f"invalid pair at line {line_no}")
        pairs.append(Pair(a=a, b=b, label=int(label)))
    if len(pairs) < 200:
        raise SystemExit(f"calibration set must contain >=200 pairs, got {len(pairs)}")
    if not any(pair.label == 1 for pair in pairs):
        raise SystemExit("calibration set must contain positive pairs")
    if not any(pair.label == 0 for pair in pairs):
        raise SystemExit("calibration set must contain negative pairs")
    return pairs


def _embed_texts(pairs: list[Pair]) -> dict[str, np.ndarray]:
    texts = list(dict.fromkeys(text for pair in pairs for text in (pair.a, pair.b)))
    embedded = embed_sync(texts)
    return {
        text: np.asarray(vector, dtype=np.float32)
        for text, vector in zip(texts, embedded, strict=True)
    }


def _scan(similarities: list[tuple[float, int]]) -> list[ScanRow]:
    positives = sum(1 for _, label in similarities if label == 1)
    negatives = len(similarities) - positives
    rows: list[ScanRow] = []
    for step in range(79):
        tau = round(0.60 + step * 0.005, 3)
        hits = sum(1 for score, label in similarities if label == 1 and score >= tau)
        false_hits = sum(1 for score, label in similarities if label == 0 and score >= tau)
        rows.append(
            ScanRow(
                tau=tau,
                hit_rate=hits / positives,
                false_hit_rate=false_hits / negatives,
                hits=hits,
                false_hits=false_hits,
            )
        )
    return rows


def _select(rows: list[ScanRow], *, max_false_hit: float) -> ScanRow:
    eligible = [row for row in rows if row.false_hit_rate <= max_false_hit]
    if not eligible:
        raise SystemExit(f"no threshold satisfies false hit <= {max_false_hit:.4f}")
    # DECISION: choose the highest recall, then the higher threshold to reduce tie risk.
    return max(eligible, key=lambda row: (row.hit_rate, row.tau))


def _operating_points(rows: list[ScanRow]) -> list[dict[str, float | str]]:
    budgets = [
        ("conservative", 0.01),
        ("balanced", 0.03),
        ("aggressive", 0.05),
    ]
    points: list[dict[str, float | str]] = []
    for name, max_fpr in budgets:
        row = _select(rows, max_false_hit=max_fpr)
        points.append(
            {
                "name": name,
                "tau": row.tau,
                "expected_fpr": round(row.false_hit_rate, 6),
                "expected_recall": round(row.hit_rate, 6),
            }
        )
    return points


def _print_report(
    *,
    pairs_path: Path,
    model: str,
    dataset_sha256: str,
    pairs: int,
    positives: int,
    negatives: int,
    rows: list[ScanRow],
    selected: ScanRow,
    operating_points: list[dict[str, float | str]],
) -> None:
    print(f"pairs={pairs} positives={positives} negatives={negatives}")
    print(f"model={model}")
    print(f"dataset={pairs_path}")
    print(f"dataset_sha256={dataset_sha256}")
    print("tau\thit_rate\tfalse_hit_rate\thits\tfalse_hits")
    for row in rows:
        print(
            f"{row.tau:.3f}\t{row.hit_rate:.4f}\t{row.false_hit_rate:.4f}"
            f"\t{row.hits}\t{row.false_hits}"
        )
    print(
        "selected"
        f"\ttau={selected.tau:.3f}"
        f"\thit_rate={selected.hit_rate:.4f}"
        f"\tfalse_hit_rate={selected.false_hit_rate:.4f}"
    )
    print("operating_points")
    for point in operating_points:
        print(
            f"{point['name']}"
            f"\ttau={float(point['tau']):.3f}"
            f"\texpected_recall={float(point['expected_recall']):.4f}"
            f"\texpected_fpr={float(point['expected_fpr']):.4f}"
        )


def _write_calibrated(
    path: Path,
    *,
    model: str,
    pairs_path: Path,
    dataset_sha256: str,
    pair_count: int,
    positives: int,
    negatives: int,
    max_false_hit: float,
    selected: ScanRow,
    operating_points: list[dict[str, float | str]],
) -> None:
    payload = {
        "cache": {
            "l2": {
                "similarity_threshold": selected.tau,
                "operating_points": operating_points,
            }
        },
        "calibration": {
            "model": model,
            "pairs": str(pairs_path),
            "pair_count": pair_count,
            "positives": positives,
            "negatives": negatives,
            "dataset_sha256": dataset_sha256,
            "generated_at": datetime.now(UTC).isoformat(),
            "max_false_hit": max_false_hit,
            "tau": selected.tau,
            "hit_rate": round(selected.hit_rate, 6),
            "false_hit_rate": round(selected.false_hit_rate, 6),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
