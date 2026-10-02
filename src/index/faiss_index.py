"""FAISS index for image search."""
from pathlib import Path
import numpy as np
import faiss
from src.config import NLIST, NPROBE


class ImageIndex:
    """FAISS IVF index over image embeddings."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.index = None
        self.metadata: list[dict] = []

    def build(self, embeddings: np.ndarray, metadata: list[dict]):
        """Train and add vectors."""
        n = embeddings.shape[0]
        # For tiny datasets use flat index
        if n < NLIST * 40:
            self.index = faiss.IndexFlatIP(self.dim)
        else:
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, NLIST, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
            self.index.nprobe = NPROBE

        self.index.add(embeddings)
        self.metadata = metadata
        return self

    def search(self, query_embedding: np.ndarray, k: int = 10) -> tuple[list[dict], list[float]]:
        if self.index is None:
            raise RuntimeError("Index not built.")
        D, I = self.index.search(query_embedding, k)
        results = [self.metadata[i] for i in I[0] if i != -1]
        scores = [float(s) for s in D[0]]
        return results, scores

    def save(self, path: str | Path):
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: str | Path, dim: int = 512, metadata: list[dict] | None = None):
        instance = cls(dim=dim)
        instance.index = faiss.read_index(str(path))
        instance.metadata = metadata or []
        return instance
