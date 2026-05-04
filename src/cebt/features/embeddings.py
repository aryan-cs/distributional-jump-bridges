"""Disclosure text embedding with deterministic cache metadata."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

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


class TextEmbedder(Protocol):
    dim: int
    model_id: str

    def encode_batch(self, texts: list[str]) -> np.ndarray: ...


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


class SentenceTransformerEmbedder:
    """Open pretrained disclosure encoder with deterministic chunk pooling."""

    def __init__(
        self,
        model_id: str = "BAAI/bge-small-en-v1.5",
        dim: int | None = None,
        batch_size: int = 16,
        max_chunk_words: int = 220,
        max_chunks: int = 12,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise ImportError(
                "Install embedding dependencies with `uv sync --extra embeddings` "
                "or run via `uv run --extra embeddings ...`."
            ) from exc
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_chunk_words = max_chunk_words
        self.max_chunks = max_chunks
        self.model = SentenceTransformer(model_id, device=device)
        model_dim = int(self.model.get_sentence_embedding_dimension())
        if dim is not None and int(dim) != model_dim:
            raise ValueError(
                f"Configured embedding dim {dim} does not match {model_id} dim {model_dim}."
            )
        self.dim = model_dim

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        flat_chunks: list[str] = []
        owners: list[int] = []
        for idx, text in enumerate(texts):
            chunks = self._chunks(text)
            flat_chunks.extend(chunks)
            owners.extend([idx] * len(chunks))
        chunk_vectors = self.model.encode(
            flat_chunks,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        output = np.zeros((len(texts), self.dim), dtype=np.float32)
        counts = np.zeros(len(texts), dtype=np.float32)
        for owner, vector in zip(owners, chunk_vectors, strict=False):
            output[owner] += vector
            counts[owner] += 1.0
        counts[counts == 0] = 1.0
        output /= counts[:, None]
        norms = np.linalg.norm(output, axis=1, keepdims=True)
        output = np.divide(output, norms, out=np.zeros_like(output), where=norms > 0)
        return output.astype(np.float32)

    def _chunks(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return [""]
        chunks = [
            " ".join(words[start : start + self.max_chunk_words])
            for start in range(0, len(words), self.max_chunk_words)
        ]
        if len(chunks) <= self.max_chunks:
            return chunks
        indices = np.linspace(0, len(chunks) - 1, self.max_chunks, dtype=int)
        return [chunks[int(index)] for index in indices]


def build_embedder(config: dict) -> TextEmbedder:
    provider = str(config.get("provider", "hashing")).lower()
    if provider == "hashing":
        return HashingEmbedder(
            dim=int(config.get("dim", 256)),
            model_id=config.get("model_id", "cebt.hashing-v1"),
        )
    if provider in {"sentence_transformers", "sentence-transformers"}:
        return SentenceTransformerEmbedder(
            model_id=config.get("model_id", "BAAI/bge-small-en-v1.5"),
            dim=config.get("dim"),
            batch_size=int(config.get("batch_size", 16)),
            max_chunk_words=int(config.get("max_chunk_words", 220)),
            max_chunks=int(config.get("max_chunks", 12)),
            device=config.get("device"),
        )
    raise ValueError(f"Unknown embedding provider: {provider}")


def load_embedding_cache(path: str | Path) -> dict[tuple[str, str, str], np.ndarray]:
    rows = read_jsonl(path)
    cache = {}
    for row in rows:
        key = (row["item_id"], row["model_id"], row["text_sha256"])
        cache[key] = np.asarray(row["embedding"], dtype=np.float32)
    return cache


def embed_texts_with_cache(
    items: list[tuple[str, str]],
    embedder: TextEmbedder,
    cache_path: str | Path,
) -> dict[str, np.ndarray]:
    cache_file = Path(cache_path)
    cache = load_embedding_cache(cache_file)
    output_rows = []
    result: dict[str, np.ndarray] = {}
    missing: list[tuple[str, str, str]] = []
    for item_id, text in items:
        text_hash = sha256_text(text)
        key = (item_id, embedder.model_id, text_hash)
        if key in cache:
            result[item_id] = cache[key]
            continue
        missing.append((item_id, text, text_hash))
    if missing:
        embeddings = embedder.encode_batch([text for _, text, _ in missing])
        for (item_id, _text, text_hash), embedding in zip(missing, embeddings, strict=True):
            embedding = np.asarray(embedding, dtype=np.float32)
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
