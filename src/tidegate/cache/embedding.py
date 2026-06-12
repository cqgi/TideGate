from __future__ import annotations

import asyncio
import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import Protocol, cast

import numpy as np


class _EmbedModel(Protocol):
    def embed(self, documents: list[str]) -> Iterable[object]: ...


class _RerankModel(Protocol):
    def rerank_pairs(
        self, pairs: list[tuple[str, str]], batch_size: int = 64
    ) -> Iterable[float]: ...


_MODEL: _EmbedModel | None = None
_RERANKER: _RerankModel | None = None


def init_embedding_worker(
    model_name: str,
    cache_dir: str,
    hf_endpoint: str,
    reranker_model: str | None = None,
) -> None:
    global _MODEL, _RERANKER
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
    os.environ.setdefault("HF_HOME", cache_dir)
    from fastembed import TextEmbedding

    # DECISION: M4 keeps the fastembed model inside the embedding pool worker so
    # requests do not serialize model state or reload weights per call.
    _MODEL = cast(_EmbedModel, TextEmbedding(model_name=model_name, cache_dir=cache_dir))
    _RERANKER = None
    if reranker_model is not None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        # DECISION: SPEC-F-2 colocates reranker and embedding models in the same L2
        # process pool so the query timeout covers both recall embedding and reranking.
        _RERANKER = cast(
            _RerankModel,
            TextCrossEncoder(model_name=reranker_model, cache_dir=cache_dir),
        )


def embed_sync(texts: list[str]) -> list[list[float]]:
    if _MODEL is None:
        raise RuntimeError("embedding worker was not initialized")
    vectors = [np.asarray(vector, dtype=np.float32) for vector in _MODEL.embed(texts)]
    return [_normalize(vector).tolist() for vector in vectors]


def rerank_sync(pairs: list[tuple[str, str]]) -> list[float]:
    if _RERANKER is None:
        raise RuntimeError("reranker worker was not initialized")
    return [float(score) for score in _RERANKER.rerank_pairs(pairs)]


class EmbeddingService:
    def __init__(self, pool: ProcessPoolExecutor) -> None:
        self._pool = pool

    async def embed(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, embed_sync, texts)

    async def rerank(self, pairs: list[tuple[str, str]]) -> list[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, rerank_sync, pairs)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm
