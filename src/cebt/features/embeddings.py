"""Disclosure text embedding with deterministic cache metadata."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from cebt.utils.hashing import sha256_text
from cebt.utils.io import read_jsonl, write_jsonl


@dataclass(frozen=True)
class CachedEmbedding:
    item_id: str
    model_id: str
    text_sha256: str
    embedding: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


class HashingEmbedder:
    """Open, deterministic fallback encoder.

    This is a real baseline embedding method, not a generated model output. It keeps
    the pipeline reproducible when larger open encoders are unavailable.
    """

    def __init__(self, dim: int = 256, model_id: str = "cebt.hashing-v1") -> None:
        self.dim = dim
        self.model_id = model_id

    def encode(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            digest = int(sha256_text(token)[:16], 16)
            index = digest % self.dim
            sign = 1.0 if (digest >> 8) % 2 == 0 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.encode(text) for text in texts], axis=0)


def load_embedding_cache(path: str | Path) -> dict[tuple[str, str, str], np.ndarray]:
    rows = read_jsonl(path)
    cache = {}
    for row in rows:
        key = (row["item_id"], row["model_id"], row["text_sha256"])
        cache[key] = np.asarray(row["embedding"], dtype=np.float32)
    return cache


def embed_texts_with_cache(
    items: list[tuple[str, str]],
    embedder: HashingEmbedder,
    cache_path: str | Path,
) -> dict[str, np.ndarray]:
    cache_file = Path(cache_path)
    cache = load_embedding_cache(cache_file)
    output_rows = []
    result: dict[str, np.ndarray] = {}
    for item_id, text in items:
        text_hash = sha256_text(text)
        key = (item_id, embedder.model_id, text_hash)
        if key in cache:
            result[item_id] = cache[key]
            continue
        embedding = embedder.encode(text)
        result[item_id] = embedding
        output_rows.append(
            CachedEmbedding(item_id, embedder.model_id, text_hash, embedding.tolist()).to_dict()
        )
    if output_rows:
        existing = read_jsonl(cache_file)
        write_jsonl(cache_file, [*existing, *output_rows])
    return result


def zero_embedding(dim: int) -> np.ndarray:
    return np.zeros(dim, dtype=np.float32)


def finite_embedding(value: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(value))) and math.isfinite(float(np.linalg.norm(value)))
