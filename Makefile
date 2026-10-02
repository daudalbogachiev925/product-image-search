.PHONY: install build-index search test serve

install:
	pip install -r requirements.txt

build-index:
	python scripts/build_index.py

search:
	python scripts/search.py

test:
	pytest tests/ -v

serve:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
