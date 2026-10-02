# Product Image Search

Vector search for products by image or text. Built on CLIP + FAISS.

## Motivation

Customers often can't describe what they want. They take a photo of a friend's sneakers or type "that red phone from the ad". This project lets you search a catalog by either modality — image or text — using CLIP's shared embedding space.

## How it works

- Image query -> CLIP encoder -> 512-dim embedding
- Text query -> CLIP encoder -> 512-dim embedding
- Both go into the same FAISS index
- Index returns top-k most similar products

Both images and text map to the same 512-dim space, so cross-modal search works out of the box.

## Quick Start

pip install -r requirements.txt

python scripts/build_index.py

python scripts/search.py

uvicorn src.serving.api:app --port 8000

## API

Search by image:

curl -X POST http://localhost:8000/search -F "file=@sneaker.jpg" -F "k=10"

Search by text:

curl -X POST http://localhost:8000/search_text -H "Content-Type: application/json" -d '{"query": "red sneakers", "k": 5}'

## Index details

- Encoder: openai/clip-vit-base-patch32 (512d, ~150M params)
- Index type: FAISS IVF for >4000 items, flat for smaller catalogs
- Similarity: cosine (via inner product on L2-normalized embeddings)

## Data

Place your product images in data/raw/product_images/. Filenames without extension become the category. Falls back to synthetic colored-shape images if missing.

## Project layout

src/
  data/
  models/
  index/
  serving/
scripts/
  build_index.py
  search.py
tests/

## Roadmap

- [x] CLIP encoder
- [x] FAISS index
- [x] Image + text search
- [ ] Hybrid search with attribute filters (category, price)
- [ ] Reranker on top of retrieval
- [ ] ONNX export for CPU-only deployment

## Known issues

- No incremental index updates (rebuild required for new items)
- CLIP struggles with fine-grained similarity (e.g. two very similar sneakers)
- No query expansion for text search
- CLIP text encoding is English-centric; multilingual queries need a different model (e.g. multilingual-clip)

## Stack

PyTorch, Transformers (CLIP), FAISS, FastAPI, Pillow

## License

MIT
