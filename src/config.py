"""Project configuration."""
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

SEED = 42

# CLIP
CLIP_MODEL = "openai/clip-vit-base-patch32"
EMBED_DIM = 512

# FAISS
FAISS_INDEX_TYPE = "IVF"
NLIST = 100
NPROBE = 10   # higher = better recall, slower search; tuned on 200-image catalog
