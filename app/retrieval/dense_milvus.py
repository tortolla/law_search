from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from pymilvus import Collection, connections, utility
from sentence_transformers import SentenceTransformer

from app.retrieval.config import (
    FRIDA_MODEL_PATH,
    MILVUS_COLLECTION,
    MILVUS_HOST,
    MILVUS_PORT,
)


_MODEL: SentenceTransformer | None = None
_CONNECTED = False
_COLLECTION_CACHE: dict[str, Collection] = {}


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_frida_model(model_path: str | Path = FRIDA_MODEL_PATH) -> SentenceTransformer:
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    model_path = Path(model_path)

    if not model_path.exists():
        _fail(f"FRIDA model directory not found: {model_path}")

    device = _choose_device()
    _MODEL = SentenceTransformer(str(model_path), device=device)

    return _MODEL


def encode_query(
    query: str,
    model_path: str | Path = FRIDA_MODEL_PATH,
    prompt_name: str = "search_query",
) -> list[float]:
    model = get_frida_model(model_path)

    emb = model.encode(
        [query],
        prompt_name=prompt_name,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    return emb.astype("float32").tolist()


def connect_milvus(
    host: str = MILVUS_HOST,
    port: str | int = MILVUS_PORT,
) -> None:
    global _CONNECTED

    if _CONNECTED:
        return

    connections.connect(alias="default", host=host, port=str(port))
    _CONNECTED = True


def get_collection(
    collection_name: str = MILVUS_COLLECTION,
    host: str = MILVUS_HOST,
    port: str | int = MILVUS_PORT,
) -> Collection:
    connect_milvus(host=host, port=port)

    if not utility.has_collection(collection_name):
        _fail(f"Milvus collection not found: {collection_name}")

    if collection_name in _COLLECTION_CACHE:
        return _COLLECTION_CACHE[collection_name]

    collection = Collection(collection_name)
    collection.load()

    _COLLECTION_CACHE[collection_name] = collection
    return collection


def search_milvus_dense(
    query: str,
    top_k: int = 10,
    collection_name: str = MILVUS_COLLECTION,
    host: str = MILVUS_HOST,
    port: str | int = MILVUS_PORT,
    prompt_name: str = "search_query",
    metric_type: str = "IP",
    ef: int = 128,
) -> pd.DataFrame:
    """
    Dense FRIDA search through Milvus.

    Returns a DataFrame compatible with the old retrieval layer:
    chunk_id, doc_id, category, source_group, title, chunk_ix, score.
    """

    collection = get_collection(
        collection_name=collection_name,
        host=host,
        port=port,
    )

    query_vector = encode_query(
        query=query,
        prompt_name=prompt_name,
    )

    search_params: dict[str, Any] = {
        "metric_type": metric_type,
        "params": {
            "ef": ef,
        },
    }

    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=[
            "chunk_id",
            "doc_id",
            "category",
            "source_group",
            "title",
            "chunk_ix",
        ],
    )

    rows: list[dict[str, Any]] = []

    for hit in results[0]:
        entity = hit.entity

        rows.append(
            {
                "chunk_id": entity.get("chunk_id"),
                "doc_id": entity.get("doc_id"),
                "category": entity.get("category"),
                "source_group": entity.get("source_group"),
                "title": entity.get("title"),
                "chunk_ix": entity.get("chunk_ix"),
                "score": float(hit.score),
                "dense_backend": "milvus",
                "collection": collection_name,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    query = "норматив стоимости одного квадратного метра общей площади жилого помещения"
    collection = sys.argv[1] if len(sys.argv) > 1 else MILVUS_COLLECTION

    df = search_milvus_dense(
        query=query,
        top_k=5,
        collection_name=collection,
    )

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
