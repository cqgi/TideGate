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

from tidegate.cache.embedding import embed_sync, init_embedding_worker, rerank_sync


@dataclass(frozen=True)
class Pair:
    a: str
    b: str
    label: int


@dataclass(frozen=True)
class ScanRow:
    tau: float
    recall: float
    fpr: float
    hits: int
    false_hits: int


@dataclass(frozen=True)
class CalibrationResult:
    name: str
    recall_threshold: float
    row: ScanRow


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/calibration/semcache_pairs.jsonl")
    parser.add_argument("--max-false-hit", type=float, default=0.01)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--model-cache-dir", default=".cache/huggingface")
    parser.add_argument("--hf-endpoint", default="")
    parser.add_argument("--recall-top-k", type=int, default=3)
    parser.add_argument("--output", default="config/calibrated.yaml")
    parser.add_argument("--json-output", default="out/calibration.json")
    args = parser.parse_args(argv)

    pairs_path = Path(args.pairs)
    raw = pairs_path.read_bytes()
    pairs = _load_pairs(raw)
    dataset_sha256 = hashlib.sha256(raw).hexdigest()

    init_embedding_worker(
        args.model,
        args.model_cache_dir,
        args.hf_endpoint,
        args.reranker_model,
    )
    vectors = _embed_texts(pairs)
    bi_scores = [float(np.dot(vectors[pair.a], vectors[pair.b])) for pair in pairs]
    single_stage_rows = _scan_thresholds(
        bi_scores, pairs, thresholds=_threshold_grid(0.50, 0.98, 0.005)
    )
    recall_threshold = _select_recall_threshold(
        bi_scores,
        pairs,
        target_recall=args.target_recall,
    )
    recall_passed = [score >= recall_threshold for score in bi_scores]
    rerank_scores = _rerank_passed_pairs(pairs, recall_passed)
    rerank_rows = _scan_rerank_thresholds(
        rerank_scores,
        pairs,
        recall_passed,
        thresholds=_rerank_threshold_grid(rerank_scores),
    )

    selected = _select(rerank_rows, max_false_hit=args.max_false_hit)
    single_points = _operating_points(
        single_stage_rows,
        recall_threshold=0.0,
        budgets=_budgets(),
    )
    rerank_points = _operating_points(
        rerank_rows,
        recall_threshold=recall_threshold,
        budgets=_budgets(),
    )
    positives = sum(1 for pair in pairs if pair.label == 1)
    negatives = len(pairs) - positives

    payload = _payload(
        pairs_path=pairs_path,
        dataset_sha256=dataset_sha256,
        pairs=pairs,
        model=args.model,
        reranker_model=args.reranker_model,
        recall_top_k=args.recall_top_k,
        target_recall=args.target_recall,
        max_false_hit=args.max_false_hit,
        recall_threshold=recall_threshold,
        selected=selected,
        single_points=single_points,
        rerank_points=rerank_points,
        single_stage_rows=single_stage_rows,
        rerank_rows=rerank_rows,
    )
    _print_report(payload, positives=positives, negatives=negatives)
    _write_calibrated(
        Path(args.output),
        payload=payload,
    )
    _write_json(Path(args.json_output), payload)


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


def _threshold_grid(start: float, stop: float, step: float) -> list[float]:
    count = round((stop - start) / step)
    return [round(start + index * step, 3) for index in range(count + 1)]


def _rerank_threshold_grid(scores: list[float | None]) -> list[float]:
    present = [score for score in scores if score is not None]
    if not present:
        return [0.0]
    low = float(np.floor(min(present) * 20) / 20)
    high = float(np.ceil(max(present) * 20) / 20)
    return _threshold_grid(low, high, 0.05)


def _scan_thresholds(
    scores: list[float],
    pairs: list[Pair],
    *,
    thresholds: list[float],
) -> list[ScanRow]:
    positives = sum(1 for pair in pairs if pair.label == 1)
    negatives = len(pairs) - positives
    rows: list[ScanRow] = []
    for tau in thresholds:
        hits = sum(
            1 for score, pair in zip(scores, pairs, strict=True) if pair.label == 1 and score >= tau
        )
        false_hits = sum(
            1 for score, pair in zip(scores, pairs, strict=True) if pair.label == 0 and score >= tau
        )
        rows.append(
            ScanRow(
                tau=tau,
                recall=hits / positives,
                fpr=false_hits / negatives,
                hits=hits,
                false_hits=false_hits,
            )
        )
    return rows


def _select_recall_threshold(
    scores: list[float],
    pairs: list[Pair],
    *,
    target_recall: float,
) -> float:
    positive_scores = sorted(score for score, pair in zip(scores, pairs, strict=True) if pair.label)
    if not positive_scores:
        raise SystemExit("cannot select recall threshold without positives")
    index = max(0, int(np.floor((1.0 - target_recall) * len(positive_scores))))
    return round(float(positive_scores[index]), 6)


def _rerank_passed_pairs(pairs: list[Pair], recall_passed: list[bool]) -> list[float | None]:
    pair_inputs = [
        (pair.a, pair.b) for pair, passed in zip(pairs, recall_passed, strict=True) if passed
    ]
    scores = rerank_sync(pair_inputs) if pair_inputs else []
    output: list[float | None] = []
    score_index = 0
    for passed in recall_passed:
        if passed:
            output.append(scores[score_index])
            score_index += 1
        else:
            output.append(None)
    return output


def _scan_rerank_thresholds(
    rerank_scores: list[float | None],
    pairs: list[Pair],
    recall_passed: list[bool],
    *,
    thresholds: list[float],
) -> list[ScanRow]:
    positives = sum(1 for pair in pairs if pair.label == 1)
    negatives = len(pairs) - positives
    rows: list[ScanRow] = []
    for tau in thresholds:
        hits = 0
        false_hits = 0
        for score, pair, passed in zip(rerank_scores, pairs, recall_passed, strict=True):
            if not passed or score is None or score < tau:
                continue
            if pair.label == 1:
                hits += 1
            else:
                false_hits += 1
        rows.append(
            ScanRow(
                tau=round(tau, 3),
                recall=hits / positives,
                fpr=false_hits / negatives,
                hits=hits,
                false_hits=false_hits,
            )
        )
    return rows


def _select(rows: list[ScanRow], *, max_false_hit: float) -> ScanRow:
    eligible = [row for row in rows if row.fpr <= max_false_hit]
    if not eligible:
        raise SystemExit(f"no threshold satisfies false hit <= {max_false_hit:.4f}")
    # DECISION: choose the highest recall, then the higher threshold to reduce tie risk.
    return max(eligible, key=lambda row: (row.recall, row.tau))


def _budgets() -> list[tuple[str, float]]:
    return [
        ("conservative", 0.01),
        ("balanced", 0.03),
        ("aggressive", 0.05),
    ]


def _operating_points(
    rows: list[ScanRow],
    *,
    recall_threshold: float,
    budgets: list[tuple[str, float]],
) -> list[CalibrationResult]:
    points: list[CalibrationResult] = []
    for name, max_fpr in budgets:
        row = _select(rows, max_false_hit=max_fpr)
        points.append(CalibrationResult(name=name, recall_threshold=recall_threshold, row=row))
    return points


def _payload(
    *,
    pairs_path: Path,
    dataset_sha256: str,
    pairs: list[Pair],
    model: str,
    reranker_model: str,
    recall_top_k: int,
    target_recall: float,
    max_false_hit: float,
    recall_threshold: float,
    selected: ScanRow,
    single_points: list[CalibrationResult],
    rerank_points: list[CalibrationResult],
    single_stage_rows: list[ScanRow],
    rerank_rows: list[ScanRow],
) -> dict[str, object]:
    positives = sum(1 for pair in pairs if pair.label == 1)
    negatives = len(pairs) - positives
    return {
        "calibration": {
            "generated_at": datetime.now(UTC).isoformat(),
            "pairs": str(pairs_path),
            "pair_count": len(pairs),
            "positives": positives,
            "negatives": negatives,
            "dataset_sha256": dataset_sha256,
            "embedding_model": model,
            "reranker_model": reranker_model,
            "recall_top_k": recall_top_k,
            "target_recall": target_recall,
            "max_false_hit": max_false_hit,
            "recall_threshold": recall_threshold,
            "selected_rerank_tau": selected.tau,
            "selected_recall": round(selected.recall, 6),
            "selected_fpr": round(selected.fpr, 6),
        },
        "comparison": [_comparison_row("single-stage bi-encoder", point) for point in single_points]
        + [_comparison_row("recall+rerank", point) for point in rerank_points],
        "single_stage_curve": [_row(row) for row in single_stage_rows],
        "rerank_curve": [_row(row) for row in rerank_rows],
    }


def _comparison_row(method: str, result: CalibrationResult) -> dict[str, object]:
    return {
        "method": method,
        "operating_point": result.name,
        "recall_threshold": round(result.recall_threshold, 6),
        "tau": result.row.tau,
        "expected_recall": round(result.row.recall, 6),
        "expected_fpr": round(result.row.fpr, 6),
        "hits": result.row.hits,
        "false_hits": result.row.false_hits,
    }


def _row(row: ScanRow) -> dict[str, float | int]:
    return {
        "tau": row.tau,
        "recall": round(row.recall, 6),
        "fpr": round(row.fpr, 6),
        "hits": row.hits,
        "false_hits": row.false_hits,
    }


def _print_report(payload: dict[str, object], *, positives: int, negatives: int) -> None:
    calibration = payload["calibration"]
    assert isinstance(calibration, dict)
    print("# Semantic Cache Calibration")
    print()
    print(
        f"pairs={calibration['pair_count']} positives={positives} negatives={negatives} "
        f"dataset_sha256={calibration['dataset_sha256']}"
    )
    print(f"embedding_model={calibration['embedding_model']}")
    print(f"reranker_model={calibration['reranker_model']}")
    print(
        f"recall_threshold={float(calibration['recall_threshold']):.6f} "
        f"target_recall={float(calibration['target_recall']):.3f}"
    )
    print()
    print("## Two-stage comparison")
    print()
    print("| method | point | recall_threshold | tau | expected_recall | expected_fpr |")
    print("|---|---|---:|---:|---:|---:|")
    comparison = payload["comparison"]
    assert isinstance(comparison, list)
    for row in comparison:
        assert isinstance(row, dict)
        print(
            f"| {row['method']} | {row['operating_point']} | "
            f"{float(row['recall_threshold']):.6f} | {float(row['tau']):.3f} | "
            f"{float(row['expected_recall']):.6f} | {float(row['expected_fpr']):.6f} |"
        )
    print()


def _write_calibrated(path: Path, *, payload: dict[str, object]) -> None:
    calibration = payload["calibration"]
    comparison = payload["comparison"]
    assert isinstance(calibration, dict)
    assert isinstance(comparison, list)
    operating_points = []
    for row in comparison:
        assert isinstance(row, dict)
        if row["method"] != "recall+rerank":
            continue
        operating_points.append(
            {
                "name": row["operating_point"],
                "tau": row["tau"],
                "expected_fpr": row["expected_fpr"],
                "expected_recall": row["expected_recall"],
            }
        )
    output = {
        "cache": {
            "l2": {
                "similarity_threshold": calibration["selected_rerank_tau"],
                "recall_top_k": calibration["recall_top_k"],
                "recall_threshold": calibration["recall_threshold"],
                "reranker_model": calibration["reranker_model"],
                "operating_points": operating_points,
            }
        },
        "calibration": calibration,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(output, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
