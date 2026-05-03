from __future__ import annotations

import numpy as np

from cebt.features.embeddings import HashingEmbedder, embed_texts_with_cache


def test_embedding_cache_reproduces_identical_vectors(tmp_path) -> None:
    embedder = HashingEmbedder(dim=16)
    cache = tmp_path / "cache.jsonl"
    first = embed_texts_with_cache([("x", "material agreement revenue")], embedder, cache)
    second = embed_texts_with_cache([("x", "material agreement revenue")], embedder, cache)
    assert np.allclose(first["x"], second["x"])
