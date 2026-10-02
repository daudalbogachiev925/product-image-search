"""FastAPI endpoint for image search."""
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
from PIL import Image
import io
import numpy as np
from pathlib import Path
from src.models.clip_encoder import CLIPEncoder
from src.index.faiss_index import ImageIndex

app = FastAPI(title="Product Image Search", version="0.1.0")

encoder: CLIPEncoder | None = None
index: ImageIndex | None = None

INDEX_PATH = Path("models/image_index.faiss")


@app.on_event("startup")
def startup():
    global encoder, index
    encoder = CLIPEncoder()
    if INDEX_PATH.exists():
        index = ImageIndex.load(INDEX_PATH)
        print(f"Index loaded with {len(index.metadata)} items")
    else:
        print(f"WARNING: no index at {INDEX_PATH}. /search will fail.")


class SearchResult(BaseModel):
    id: int
    category: str
    score: float


@app.get("/health")
def health():
    return {"status": "ok", "index_loaded": index is not None}


@app.post("/search", response_model=List[SearchResult])
async def search_image(file: UploadFile = File(...), k: int = 10):
    if index is None:
        raise HTTPException(503, "Index not loaded")
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    emb = encoder.encode_images([img])
    results, scores = index.search(emb, k=k)
    return [SearchResult(id=r["id"], category=r["category"], score=s) for r, s in zip(results, scores)]


class TextSearchRequest(BaseModel):
    query: str
    k: int = 10


@app.post("/search_text", response_model=List[SearchResult])
def search_text(req: TextSearchRequest):
    if index is None:
        raise HTTPException(503, "Index not loaded")
    emb = encoder.encode_text([req.query])
    results, scores = index.search(emb, k=req.k)
    return [SearchResult(id=r["id"], category=r["category"], score=s) for r, s in zip(results, scores)]
