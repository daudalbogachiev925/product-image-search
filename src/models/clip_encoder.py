"""CLIP encoder for image and text embeddings."""
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from src.config import CLIP_MODEL


class CLIPEncoder:
    """
    Wraps CLIP for producing image/text embeddings in the same space.
    Use: it lets you search by image OR by text query.
    """

    def __init__(self, model_name: str = CLIP_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image], batch_size: int = 16) -> "np.ndarray":
        import numpy as np
        all_embs = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            embs = self.model.get_image_features(**inputs)
            embs = embs / embs.norm(dim=-1, keepdim=True)
            all_embs.append(embs.cpu().numpy())
        return np.vstack(all_embs).astype("float32")

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> "np.ndarray":
        import numpy as np
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        embs = self.model.get_text_features(**inputs)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        return embs.cpu().numpy().astype("float32")
