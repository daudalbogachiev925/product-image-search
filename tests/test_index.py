"""Tests for FAISS index wrapper."""
import numpy as np
from src.index.faiss_index import ImageIndex


def test_build_and_search():
    rng = np.random.default_rng(0)
    embs = rng.normal(0, 1, size=(50, 16)).astype("float32")
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    meta = [{"id": i, "category": f"cat_{i % 3}"} for i in range(50)]

    index = ImageIndex(dim=16).build(embs, meta)

    # Search with the first vector — should return itself first
    results, scores = index.search(embs[:1], k=3)
    assert results[0]["id"] == 0
    assert scores[0] > 0.9


def test_index_raises_before_build():
    index = ImageIndex(dim=16)
    try:
        index.search(np.zeros((1, 16), dtype="float32"))
        assert False, "Should have raised"
    except RuntimeError:
        pass
