"""Load product images and generate synthetic data if missing."""
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image, ImageDraw
from src.config import DATA_DIR, SEED

# Categories used to color-code synthetic products
CATEGORIES = ["shoes", "shirt", "phone", "laptop", "book", "mug"]


def load_product_images(n: int = 200) -> tuple[list[Image.Image], list[dict]]:
    """
    Load images from data/raw/product_images/, else generate synthetic.

    In production: images from your product catalog (Ozon/WB).
    """
    real_dir = DATA_DIR / "raw" / "product_images"
    if real_dir.exists() and any(real_dir.iterdir()):
        images, metadata = [], []
        for i, p in enumerate(sorted(real_dir.glob("*.jpg"))[:n]):
            images.append(Image.open(p).convert("RGB"))
            metadata.append({"id": i, "path": str(p), "category": p.stem})
        return images, metadata

    print("Real images not found, generating synthetic.")
    return _generate_synthetic(n)


def _generate_synthetic(n: int) -> tuple[list[Image.Image], list[dict]]:
    """Generate synthetic product images: colored shapes on white background."""
    rng = np.random.default_rng(SEED)
    images, metadata = [], []

    for i in range(n):
        cat = CATEGORIES[i % len(CATEGORIES)]
        img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Category-specific colors and shapes
        if cat == "shoes":
            color = (150 + rng.integers(0, 100), 50, 50)
            draw.ellipse([40, 120, 180, 180], fill=tuple(int(c) for c in color))
        elif cat == "shirt":
            color = (50, 100 + rng.integers(0, 100), 150)
            draw.rectangle([60, 60, 160, 180], fill=tuple(int(c) for c in color))
        elif cat == "phone":
            color = (30, 30, 30)
            draw.rectangle([80, 40, 140, 190], fill=tuple(int(c) for c in color))
        elif cat == "laptop":
            color = (100, 100, 100)
            draw.rectangle([50, 80, 170, 150], fill=tuple(int(c) for c in color))
        elif cat == "book":
            color = (rng.integers(100, 200), rng.integers(50, 150), rng.integers(50, 150))
            draw.rectangle([70, 60, 150, 180], fill=tuple(int(c) for c in color))
        else:  # mug
            color = (200, 150, 100)
            draw.ellipse([80, 80, 150, 160], fill=tuple(int(c) for c in color))

        # Add random noise
        arr = np.array(img).astype(np.int16)
        arr += rng.integers(-15, 15, arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

        images.append(img)
        metadata.append({"id": i, "category": cat})

    return images, metadata
