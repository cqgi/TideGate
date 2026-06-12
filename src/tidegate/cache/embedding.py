from __future__ import annotations

import asyncio
import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import Protocol, cast

import numpy as np


class _EmbedModel(Protocol):
    def embed(self, documents: list[str]) -> Iterable[object]: ...


_MODEL: _EmbedModel | None = None


def init_embedding_worker(model_name: str, cache_dir: str, hf_endpoint: str) -> None:
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
    os.environ.setdefault("HF_HOME", cache_dir)
    from fastembed import TextEmbedding

    global _MODEL
    # DECISION: M4 keeps the fastembed model inside the embedding pool worker so
    # requests do not serialize model state or reload weights per call.
    _MODEL = cast(_EmbedModel, TextEmbedding(model_name=model_name, cache_dir=cache_dir))


def embed_sync(texts: list[str]) -> list[list[float]]:
    if _MODEL is None:
        raise RuntimeError("embedding worker was not initialized")
    vectors = [np.asarray(vector, dtype=np.float32) for vector in _MODEL.embed(texts)]
    return [_normalize(vector).tolist() for vector in vectors]


class EmbeddingService:
    def __init__(self, pool: ProcessPoolExecutor) -> None:
        self._pool = pool

    async def embed(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, embed_sync, texts)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm
