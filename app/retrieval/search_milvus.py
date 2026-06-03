from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from pymilvus import Collection, connections, utility
from sentence_transformers import SentenceTransformer

from app.retrieval.config import (
    BM25_PATH,
    CHUNKS_PATH,
    FRIDA_MODEL_PATH,
    MILVUS_COLLECTION,
    MILVUS_HOST,
    MILVUS_PORT,
    DEFAULT_BM25_WEIGHT,
    DEFAULT_FRIDA_WEIGHT,
    DEFAULT_CANDIDATE_K,
)
from app.retrieval.utils import tokenize_bm25


_MODEL: SentenceTransformer | None = None
_CHUNKS_DF: pd.DataFrame | None = None
_BM25_PAYLOAD: dict[str, Any] | None = None
_CONNECTED = False
_COLLECTIONS: dict[str, Collection] = {}


def _choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    path = Path(FRIDA_MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(f"FRIDA model not found: {path}")

    _MODEL = SentenceTransformer(str(path), device=_choose_device())
    return _MODEL


def _load_chunks() -> pd.DataFrame:
    global _CHUNKS_DF
    if _CHUNKS_DF is not None:
        return _CHUNKS_DF

    path = Path(CHUNKS_PATH)
    if not path.exists():
        raise FileNotFoundError(f"chunks.parquet not found: {path}")

    df = pd.read_parquet(path)
    df["chunk_id"] = df["chunk_id"].astype(str)
    df["doc_id"] = df["doc_id"].astype(str)

    _CHUNKS_DF = df
    return _CHUNKS_DF


def _load_bm25() -> dict[str, Any]:
    global _BM25_PAYLOAD
    if _BM25_PAYLOAD is not None:
        return _BM25_PAYLOAD

    path = Path(BM25_PATH)
    if not path.exists():
        raise FileNotFoundError(f"bm25.pkl not found: {path}")

    with open(path, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict) or "bm25" not in payload:
        raise ValueError("Invalid BM25 payload: expected dict with key 'bm25'")

    _BM25_PAYLOAD = payload
    return _BM25_PAYLOAD


def _encode_query(query: str) -> list[float]:
    model = _load_model()
    emb = model.encode(
        [query],
        prompt_name="search_query",
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return emb.astype("float32").tolist()


def _connect(host: str, port: str | int) -> None:
    global _CONNECTED
    if _CONNECTED:
        return
    connections.connect(alias="default", host=host, port=str(port))
    _CONNECTED = True


def _get_collection(collection_name: str, host: str, port: str | int) -> Collection:
    _connect(host, port)

    if not utility.has_collection(collection_name):
        raise RuntimeError(f"Milvus collection not found: {collection_name}")

    if collection_name in _COLLECTIONS:
        return _COLLECTIONS[collection_name]

    c = Collection(collection_name)
    c.load()
    _COLLECTIONS[collection_name] = c
    return c


def _minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype("float64", copy=False)
    if len(x) == 0:
        return x
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype="float64")
    return (x - mn) / (mx - mn)


def _bm25_candidates(query: str, candidate_k: int) -> pd.DataFrame:
    chunks = _load_chunks()
    payload = _load_bm25()
    bm25 = payload["bm25"]

    tokens = tokenize_bm25(query)
    scores = np.asarray(bm25.get_scores(tokens), dtype="float64")

    if len(scores) != len(chunks):
        raise ValueError(f"BM25 scores length != chunks rows: {len(scores)} != {len(chunks)}")

    k = min(candidate_k, len(scores))
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]

    out = chunks.iloc[idx].copy()
    out["bm25_score_raw"] = scores[idx]
    out["bm25_score"] = _minmax(out["bm25_score_raw"].to_numpy())
    return out


def _milvus_candidates(
    query: str,
    candidate_k: int,
    collection_name: str,
    host: str,
    port: str | int,
) -> pd.DataFrame:
    c = _get_collection(collection_name, host, port)
    qvec = _encode_query(query)

    res = c.search(
        data=[qvec],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"ef": max(128, int(candidate_k))}},
        limit=candidate_k,
        output_fields=["chunk_id", "doc_id", "category", "source_group", "title", "chunk_ix"],
    )

    rows = []
    for hit in res[0]:
        e = hit.entity
        rows.append(
            {
                "chunk_id": str(e.get("chunk_id")),
                "doc_id": str(e.get("doc_id")),
                "category": e.get("category"),
                "source_group": e.get("source_group"),
                "title": e.get("title"),
                "chunk_ix": int(e.get("chunk_ix")),
                "frida_score_raw": float(hit.score),
            }
        )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    df["frida_score"] = _minmax(df["frida_score_raw"].to_numpy())
    return df


def search_bm25_milvus_weighted(
    query: str,
    top_k: int = 5,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    frida_weight: float = DEFAULT_FRIDA_WEIGHT,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    collection_name: str = MILVUS_COLLECTION,
    host: str = MILVUS_HOST,
    port: str | int = MILVUS_PORT,
) -> pd.DataFrame:
    query = query.strip()
    if not query:
        raise ValueError("Empty query")

    chunks = _load_chunks()

    bm25_df = _bm25_candidates(query, candidate_k=candidate_k)
    dense_df = _milvus_candidates(
        query=query,
        candidate_k=candidate_k,
        collection_name=collection_name,
        host=host,
        port=port,
    )

    base_cols = ["chunk_id", "doc_id", "category", "source_group", "title", "chunk_ix"]

    bm25_small = bm25_df[base_cols + ["bm25_score", "bm25_score_raw"]].copy()

    if len(dense_df) == 0:
        merged = bm25_small.copy()
        merged["frida_score"] = 0.0
        merged["frida_score_raw"] = 0.0
    else:
        dense_small = dense_df[base_cols + ["frida_score", "frida_score_raw"]].copy()
        merged = pd.merge(bm25_small, dense_small, on=base_cols, how="outer")

    merged["bm25_score"] = merged["bm25_score"].fillna(0.0)
    merged["frida_score"] = merged["frida_score"].fillna(0.0)
    merged["bm25_score_raw"] = merged["bm25_score_raw"].fillna(0.0)
    merged["frida_score_raw"] = merged["frida_score_raw"].fillna(0.0)

    merged["hybrid_score"] = (
        float(bm25_weight) * merged["bm25_score"]
        + float(frida_weight) * merged["frida_score"]
    )

    merged = merged.sort_values("hybrid_score", ascending=False).head(top_k).copy()

    meta_cols = [
        "chunk_id",
        "txt_path",
        "meta_file",
        "chunk_text",
        "md_path",
        "url",
        "section_code",
        "section_title",
        "key_words_text",
        "hierarchy_chain",
    ]
    available = [c for c in meta_cols if c in chunks.columns]
    meta = chunks[available].copy()

    merged = pd.merge(merged, meta, on="chunk_id", how="left")

    merged["method"] = "bm25_milvus_weighted"
    merged["dense_backend"] = "milvus"
    merged["collection"] = collection_name

    return merged.reset_index(drop=True)
