"""Build FAISS index from product images."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_product_images
from src.models.clip_encoder import CLIPEncoder
from src.index.faiss_index import ImageIndex
from src.config import MODELS_DIR


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    print("Loading images...")
    images, metadata = load_product_images(n=200)
    print(f"  {len(images)} images, categories: {set(m['category'] for m in metadata)}")

    print("Encoding with CLIP...")
    encoder = CLIPEncoder()
    embs = encoder.encode_images(images, batch_size=16)
    print(f"  Embeddings shape: {embs.shape}")

    print("Building FAISS index...")
    index = ImageIndex(dim=embs.shape[1])
    index.build(embs, metadata)

    out_path = MODELS_DIR / "image_index.faiss"
    index.save(out_path)
    print(f"Index saved to {out_path}")

    import pickle
    with open(MODELS_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print("Metadata saved")


if __name__ == "__main__":
    main()
