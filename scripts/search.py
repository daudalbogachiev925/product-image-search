"""Demo: search by text or image."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
from src.models.clip_encoder import CLIPEncoder
from src.index.faiss_index import ImageIndex
from src.data.loader import load_product_images
from src.config import MODELS_DIR


def main():
    with open(MODELS_DIR / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    encoder = CLIPEncoder()
    index = ImageIndex.load(MODELS_DIR / "image_index.faiss", metadata=metadata)
    print(f"Index loaded: {len(metadata)} items")

    # Text search
    for query in ["red sneakers", "a phone", "wooden mug"]:
        emb = encoder.encode_text([query])
        results, scores = index.search(emb, k=5)
        print(f"\nQuery: '{query}'")
        for r, s in zip(results, scores):
            print(f"  [{s:.3f}] id={r['id']} cat={r['category']}")

    # Image search
    images, _ = load_product_images(n=5)
    query_img = images[0]
    emb = encoder.encode_images([query_img])
    results, scores = index.search(emb, k=5)
    print(f"\nImage query (first image)")
    for r, s in zip(results, scores):
        print(f"  [{s:.3f}] id={r['id']} cat={r['category']}")


if __name__ == "__main__":
    main()
