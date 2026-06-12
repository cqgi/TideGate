from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from calibrate_cache_threshold import Pair, _embed_texts, _load_pairs

from tidegate.cache.embedding import embed_sync, init_embedding_worker


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/calibration/semcache_pairs.jsonl")
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--model-cache-dir", default=".cache/huggingface")
    parser.add_argument("--hf-endpoint", default="")
    args = parser.parse_args(argv)

    pairs_path = Path(args.pairs)
    raw = pairs_path.read_bytes()
    pairs = _load_pairs(raw)
    init_embedding_worker(args.model, args.model_cache_dir, args.hf_endpoint)
    vectors = _embed_texts(pairs)
    scored = [(pair, float(np.dot(vectors[pair.a], vectors[pair.b]))) for pair in pairs]

    positives = [score for pair, score in scored if pair.label == 1]
    negatives = [score for pair, score in scored if pair.label == 0]

    print("# Semantic Cache Bi-Encoder Diagnostic")
    print()
    print(f"pairs={len(pairs)} positives={len(positives)} negatives={len(negatives)}")
    print(f"dataset={pairs_path}")
    print(f"dataset_sha256={hashlib.sha256(raw).hexdigest()}")
    print(f"model={args.model}")
    print()
    _print_percentiles("positive", positives)
    _print_percentiles("negative", negatives)
    print()
    print("## Lowest Positive Top10")
    print()
    print("| rank | score | a | b |")
    print("|---:|---:|---|---|")
    for rank, (pair, score) in enumerate(_lowest_positives(scored), start=1):
        print(f"| {rank} | {score:.6f} | {_cell(pair.a)} | {_cell(pair.b)} |")
    print()
    _print_trivial_self_check(args.model)


def _print_percentiles(name: str, values: list[float]) -> None:
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"## {name} similarity percentiles")
    print()
    print("| percentile | similarity |")
    print("|---:|---:|")
    for percentile in percentiles:
        value = float(np.percentile(values, percentile))
        print(f"| p{percentile} | {value:.6f} |")
    print()


def _lowest_positives(scored: list[tuple[Pair, float]]) -> list[tuple[Pair, float]]:
    positives = [(pair, score) for pair, score in scored if pair.label == 1]
    return sorted(positives, key=lambda item: item[1])[:10]


def _print_trivial_self_check(model: str) -> None:
    del model
    pairs = [
        ("退款流程是什么", "退款流程是什么"),
        ("退款流程是什么", "今天天气怎么样"),
        ("连续包月怎么关", "关闭连续订阅入口在哪"),
    ]
    texts = list(dict.fromkeys(text for pair in pairs for text in pair))
    vectors = {
        text: np.asarray(vector, dtype=np.float32)
        for text, vector in zip(texts, embed_sync(texts), strict=True)
    }
    print("## Trivial Pair Self-Check")
    print()
    print("| a | b | similarity |")
    print("|---|---|---:|")
    for a, b in pairs:
        score = float(np.dot(vectors[a], vectors[b]))
        print(f"| {_cell(a)} | {_cell(b)} | {score:.6f} |")


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
