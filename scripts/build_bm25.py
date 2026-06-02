import pickle

import pandas as pd
from rank_bm25 import BM25Plus
from tqdm.auto import tqdm

from app.retrieval.config import CHUNKS_PATH, BM25_DIR, BM25_PATH
from app.retrieval.utils import tokenize_bm25


def main():
    BM25_DIR.mkdir(parents=True, exist_ok=True)

    chunks_df = pd.read_parquet(CHUNKS_PATH)
    texts = chunks_df["chunk_text"].fillna("").tolist()

    corpus_tokens = [tokenize_bm25(t) for t in tqdm(texts, desc="Токенизация для BM25")]
    bm25 = BM25Plus(corpus_tokens)

    payload = {
        "bm25": bm25,
        "num_chunks": len(texts),
    }

    with open(BM25_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"[OK] BM25 index -> {BM25_PATH}")
    print(f"Chunks indexed: {len(texts)}")


if __name__ == "__main__":
    main()