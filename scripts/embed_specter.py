"""Encode every paper with SPECTER1 (title [SEP] abstract) and save the vectors to artifacts/specter.npz.
SPECTER1 was trained on citations of papers published before 2020, so it has not seen the post-2020 citations
that E2 and E2b hide. Needs extra packages that the app does not install:
    uv run --with sentence-transformers --with torch --index https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match python scripts/embed_specter.py"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
MODEL = "sentence-transformers/allenai-specter"


def main():
    with (ROOT / "data/papers.jsonl").open() as source:
        papers = [json.loads(line) for line in source if line.strip()]
    model = SentenceTransformer(MODEL, device="cpu")
    texts = [p["title"] + model.tokenizer.sep_token + (p.get("abstract") or "") for p in papers]
    vectors = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    np.savez_compressed(ROOT / "artifacts/specter.npz", ids=np.array([p["id"] for p in papers]),
                        vectors=vectors.astype(np.float16), model=MODEL)
    print(len(papers), vectors.shape)


if __name__ == "__main__":
    main()
