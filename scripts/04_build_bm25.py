from __future__ import annotations

import json
import pickle
from datetime import datetime

import pandas as pd
from rank_bm25 import BM25Plus
from tqdm.auto import tqdm

from app.retrieval.config import CHUNKS_PATH, BM25_DIR, BM25_PATH
from app.retrieval.utils import tokenize_bm25


def main() -> None:
    print("=" * 100)
    print("BUILD BM25 INDEX")
    print("=" * 100)
    print(f"CHUNKS_PATH: {CHUNKS_PATH}")
    print(f"BM25_DIR:    {BM25_DIR}")
    print(f"BM25_PATH:   {BM25_PATH}")

    BM25_DIR.mkdir(parents=True, exist_ok=True)

    chunks_df = pd.read_parquet(CHUNKS_PATH)

    required_columns = ["chunk_id", "doc_id", "chunk_text"]
    missing = [c for c in required_columns if c not in chunks_df.columns]
    if missing:
        raise ValueError(f"chunks.parquet missing required columns: {missing}")

    if not chunks_df["chunk_id"].astype(str).is_unique:
        duplicated = chunks_df["chunk_id"].astype(str).value_counts()
        duplicated = duplicated[duplicated > 1].head(20)
        print(duplicated.to_string())
        raise ValueError("chunk_id is not unique")

    texts = chunks_df["chunk_text"].fillna("").astype(str).tolist()
    chunk_ids = chunks_df["chunk_id"].astype(str).tolist()
    doc_ids = chunks_df["doc_id"].astype(str).tolist()

    print(f"Chunks loaded: {len(texts)}")

    empty_texts = sum(1 for x in texts if not x.strip())
    if empty_texts:
        print(f"[WARN] empty chunk_text rows: {empty_texts}")

    corpus_tokens = [
        tokenize_bm25(text)
        for text in tqdm(texts, desc="Токенизация для BM25")
    ]

    empty_tokenized = sum(1 for tokens in corpus_tokens if len(tokens) == 0)
    if empty_tokenized:
        print(f"[WARN] empty tokenized chunks: {empty_tokenized}")

    bm25 = BM25Plus(corpus_tokens)

    payload = {
        "bm25": bm25,
        "num_chunks": len(texts),
        "chunk_ids": chunk_ids,
        "doc_ids": doc_ids,
        "tokenizer": "app.retrieval.utils.tokenize_bm25",
        "bm25_class": "rank_bm25.BM25Plus",
        "chunks_path": str(CHUNKS_PATH),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    with open(BM25_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"[OK] BM25 index -> {BM25_PATH}")
    print(f"Chunks indexed: {len(texts)}")

    info_path = BM25_DIR / "bm25_info.json"
    info = {k: v for k, v in payload.items() if k != "bm25"}
    info_path.write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] BM25 info -> {info_path}")


if __name__ == "__main__":
    main()
