from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.retrieval.config import (
    ROOT_DIR,
    DOCS_PATH,
    CHUNKS_PATH,
    BM25_PATH,
    FRIDA_EMBEDDINGS_PATH,
    FRIDA_INFO_PATH,
    FRIDA_MODEL_PATH,
    get_gold_dataset_path,
)


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def fail(message: str) -> None:
    print(f"[ERROR] {message}")
    sys.exit(1)


def require_exists(path: Path, label: str) -> None:
    if not path.exists():
        fail(f"{label} not found: {path}")
    ok(f"{label} exists: {path}")


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        warn(f"cannot parse json {path}: {e}")
        return None


def validate_required_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        fail(f"{label} missing required columns: {missing}")
    ok(f"{label} required columns exist")


def validate_docs(docs: pd.DataFrame) -> None:
    print()
    print("DOCS VALIDATION")
    print("-" * 100)

    required = [
        "doc_id",
        "base_doc_id",
        "title",
        "category",
        "source_group",
        "md_path",
        "text",
        "doc_len",
    ]

    validate_required_columns(docs, required, "docs.parquet")

    if docs["doc_id"].isna().any():
        fail("docs.doc_id contains NaN")
    ok("docs.doc_id has no NaN")

    if not docs["doc_id"].astype(str).is_unique:
        duplicated = docs["doc_id"].astype(str).value_counts()
        duplicated = duplicated[duplicated > 1].head(20)
        print(duplicated.to_string())
        warn(
            "docs.doc_id is not unique. This is acceptable when the same official "
            "document id appears in several categories/source groups. chunk_id remains "
            "the strict unique retrieval key."
        )
    else:
        ok("docs.doc_id is unique")

    empty_text = docs["text"].fillna("").astype(str).str.len().eq(0).sum()
    if empty_text:
        warn(f"empty docs.text rows: {empty_text}")
    else:
        ok("no empty docs.text")

    if "txt_path" in docs.columns:
        empty_txt_path = docs["txt_path"].fillna("").astype(str).str.len().eq(0).sum()
        if empty_txt_path:
            warn(f"empty docs.txt_path rows: {empty_txt_path}")
        else:
            ok("no empty docs.txt_path")
    else:
        warn("docs.txt_path is absent; not fatal, but chunks.txt_path is required")


def validate_chunks(chunks: pd.DataFrame, docs: pd.DataFrame) -> None:
    print()
    print("CHUNKS VALIDATION")
    print("-" * 100)

    required = [
        "doc_id",
        "base_doc_id",
        "title",
        "category",
        "source_group",
        "md_path",
        "txt_path",
        "chunk_id",
        "chunk_ix",
        "chunk_text",
        "chunk_len",
    ]

    validate_required_columns(chunks, required, "chunks.parquet")

    if chunks["chunk_id"].isna().any():
        fail("chunks.chunk_id contains NaN")
    ok("chunks.chunk_id has no NaN")

    chunk_ids = chunks["chunk_id"].astype(str)

    if not chunk_ids.is_unique:
        duplicated = chunk_ids.value_counts()
        duplicated = duplicated[duplicated > 1].head(20)
        print(duplicated.to_string())
        fail("chunks.chunk_id is not unique")
    ok("chunks.chunk_id is unique")

    empty_chunk_text = chunks["chunk_text"].fillna("").astype(str).str.len().eq(0).sum()
    if empty_chunk_text:
        fail(f"empty chunks.chunk_text rows: {empty_chunk_text}")
    ok("no empty chunks.chunk_text")

    empty_txt_path = chunks["txt_path"].fillna("").astype(str).str.len().eq(0).sum()
    if empty_txt_path:
        fail(f"empty chunks.txt_path rows: {empty_txt_path}")
    ok("no empty chunks.txt_path")

    docs_doc_ids = set(docs["doc_id"].astype(str))
    chunks_doc_ids = set(chunks["doc_id"].astype(str))
    missing_docs = chunks_doc_ids - docs_doc_ids

    if missing_docs:
        examples = sorted(list(missing_docs))[:20]
        fail(f"chunks reference doc_id absent in docs.parquet: {len(missing_docs)}; examples={examples}")
    ok("all chunks.doc_id values exist in docs.parquet")

    bad_chunk_len = (
        chunks["chunk_len"].fillna(-1).astype(int)
        != chunks["chunk_text"].fillna("").astype(str).str.len()
    ).sum()

    if bad_chunk_len:
        warn(f"chunk_len does not match actual chunk_text length for rows: {bad_chunk_len}")
    else:
        ok("chunk_len matches chunk_text length")


def validate_embeddings(chunks: pd.DataFrame) -> np.ndarray:
    print()
    print("FRIDA EMBEDDINGS VALIDATION")
    print("-" * 100)

    require_exists(FRIDA_EMBEDDINGS_PATH, "FRIDA embeddings.npy")

    try:
        embeddings = np.load(FRIDA_EMBEDDINGS_PATH)
    except Exception as e:
        fail(f"cannot load embeddings.npy: {e}")

    ok("embeddings.npy loaded")

    print(f"embeddings shape: {embeddings.shape}")
    print(f"embeddings dtype:  {embeddings.dtype}")

    if embeddings.ndim != 2:
        fail(f"embeddings must be 2D, got shape={embeddings.shape}")
    ok("embeddings is 2D matrix")

    if embeddings.shape[0] != len(chunks):
        fail(f"embeddings rows != chunks rows: {embeddings.shape[0]} != {len(chunks)}")
    ok("embeddings rows == chunks rows")

    if not np.isfinite(embeddings[: min(len(embeddings), 1000)]).all():
        fail("non-finite values detected in embeddings sample")
    ok("embeddings sample has finite values")

    sample_size = min(len(embeddings), 1000)
    norms = np.linalg.norm(embeddings[:sample_size], axis=1)

    print(f"sample norm mean: {float(norms.mean()):.6f}")
    print(f"sample norm min:  {float(norms.min()):.6f}")
    print(f"sample norm max:  {float(norms.max()):.6f}")

    if norms.min() < 0.9 or norms.max() > 1.1:
        warn("embedding norms are not close to 1.0; check normalization setting")
    else:
        ok("embedding norms look normalized")

    if FRIDA_INFO_PATH.exists():
        info = load_json_if_exists(FRIDA_INFO_PATH)
        if info is not None:
            ok(f"model_info.json loaded: {FRIDA_INFO_PATH}")

            expected_num = info.get("num_chunks")
            expected_dim = info.get("embedding_dim")

            if expected_num is not None:
                if int(expected_num) != len(chunks):
                    fail(f"model_info num_chunks != chunks rows: {expected_num} != {len(chunks)}")
                ok("model_info num_chunks == chunks rows")

            if expected_dim is not None:
                if int(expected_dim) != embeddings.shape[1]:
                    fail(f"model_info embedding_dim != embeddings dim: {expected_dim} != {embeddings.shape[1]}")
                ok("model_info embedding_dim == embeddings dim")

            print("model_info summary:")
            for key in [
                "model_path",
                "chunks_path",
                "num_chunks",
                "embedding_dim",
                "dtype",
                "prompt_name",
                "normalized",
                "device",
                "batch_size",
                "created_at",
                "chunks_format",
            ]:
                if key in info:
                    print(f"  {key}: {info[key]}")
    else:
        warn(f"model_info.json not found: {FRIDA_INFO_PATH}")

    if FRIDA_MODEL_PATH.exists():
        ok(f"FRIDA model directory exists: {FRIDA_MODEL_PATH}")
    else:
        warn(f"FRIDA model directory not found: {FRIDA_MODEL_PATH}")

    return embeddings


def validate_bm25(chunks: pd.DataFrame) -> None:
    print()
    print("BM25 VALIDATION")
    print("-" * 100)

    require_exists(BM25_PATH, "BM25 index")

    try:
        with open(BM25_PATH, "rb") as f:
            payload = pickle.load(f)
    except Exception as e:
        fail(f"cannot load bm25.pkl: {e}")

    ok("bm25.pkl loaded")

    if not isinstance(payload, dict):
        warn(f"BM25 payload is not dict: {type(payload)}")
        warn("cannot strictly validate BM25 metadata")
        return

    print("bm25 keys:", list(payload.keys()))

    if "bm25" not in payload:
        fail("BM25 payload has no 'bm25' key")
    ok("BM25 payload has bm25 object")

    if "num_chunks" not in payload:
        fail("BM25 payload has no num_chunks")
    ok("BM25 payload has num_chunks")

    bm25_num_chunks = int(payload["num_chunks"])
    print(f"bm25 num_chunks: {bm25_num_chunks}")

    if bm25_num_chunks != len(chunks):
        fail(f"bm25 num_chunks != chunks rows: {bm25_num_chunks} != {len(chunks)}")
    ok("bm25 num_chunks == chunks rows")

    if "chunk_ids" in payload and payload["chunk_ids"] is not None:
        bm25_chunk_ids = [str(x) for x in payload["chunk_ids"]]

        if len(bm25_chunk_ids) != len(chunks):
            fail(f"bm25 chunk_ids length != chunks rows: {len(bm25_chunk_ids)} != {len(chunks)}")
        ok("bm25 chunk_ids length == chunks rows")

        chunks_chunk_ids = chunks["chunk_id"].astype(str).tolist()

        if bm25_chunk_ids[:10] != chunks_chunk_ids[:10]:
            warn("first 10 BM25 chunk_ids differ from chunks.parquet order")
            print("BM25 first 10:", bm25_chunk_ids[:10])
            print("CHUNKS first 10:", chunks_chunk_ids[:10])
        else:
            ok("BM25 chunk_ids order matches chunks.parquet sample")

        if bm25_chunk_ids != chunks_chunk_ids:
            warn("BM25 chunk_ids are not exactly equal to chunks.parquet order")
        else:
            ok("BM25 chunk_ids exactly match chunks.parquet order")
    else:
        warn("BM25 payload has no chunk_ids; rebuild with scripts/04_build_bm25.py for strict validation")

    if "doc_ids" in payload and payload["doc_ids"] is not None:
        bm25_doc_ids = [str(x) for x in payload["doc_ids"]]

        if len(bm25_doc_ids) != len(chunks):
            fail(f"bm25 doc_ids length != chunks rows: {len(bm25_doc_ids)} != {len(chunks)}")
        ok("bm25 doc_ids length == chunks rows")
    else:
        warn("BM25 payload has no doc_ids; rebuild with scripts/04_build_bm25.py for strict validation")

    info_path = BM25_PATH.parent / "bm25_info.json"
    if info_path.exists():
        ok(f"bm25_info.json exists: {info_path}")
    else:
        warn(f"bm25_info.json not found: {info_path}")


def validate_gold_dataset(docs: pd.DataFrame, chunks: pd.DataFrame) -> None:
    print()
    print("GOLD DATASET VALIDATION")
    print("-" * 100)

    gold_path = get_gold_dataset_path()

    if gold_path is None:
        warn("gold dataset not found in standard locations")
        return

    ok(f"gold dataset found: {gold_path}")

    try:
        data = json.loads(gold_path.read_text(encoding="utf-8"))
    except Exception as e:
        warn(f"cannot parse gold dataset: {e}")
        return

    if not isinstance(data, list):
        warn(f"gold dataset is not list: {type(data)}")
        return

    print(f"gold rows: {len(data)}")

    docs_doc_ids = set(docs["doc_id"].astype(str))
    chunks_chunk_ids = set(chunks["chunk_id"].astype(str))

    gold_doc_ids: list[str] = []
    gold_chunk_ids: list[str] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        for key in ["source_doc_id", "doc_id", "gold_doc_id"]:
            if item.get(key):
                gold_doc_ids.append(str(item[key]))
                break

        for key in ["source_chunk_id", "chunk_id", "gold_chunk_id"]:
            if item.get(key):
                gold_chunk_ids.append(str(item[key]))
                break

    print(f"gold doc ids:        {len(gold_doc_ids)}")
    print(f"unique gold doc ids: {len(set(gold_doc_ids))}")
    print(f"gold chunk ids:      {len(gold_chunk_ids)}")
    print(f"unique gold chunks:  {len(set(gold_chunk_ids))}")

    if gold_doc_ids:
        present_docs = sum(1 for x in gold_doc_ids if x in docs_doc_ids)
        print(f"gold docs present:   {present_docs}/{len(gold_doc_ids)}")
        if present_docs == 0:
            warn("no gold doc ids are present in docs.parquet")
        else:
            ok("some gold doc ids are present in docs.parquet")

    if gold_chunk_ids:
        present_chunks = sum(1 for x in gold_chunk_ids if x in chunks_chunk_ids)
        print(f"gold chunks present: {present_chunks}/{len(gold_chunk_ids)}")

        if present_chunks == 0:
            warn("gold chunk ids are absent in current chunks; chunk-level eval is invalid")
        elif present_chunks < len(gold_chunk_ids):
            warn("some gold chunk ids are absent in current chunks")
        else:
            ok("all gold chunk ids are present in current chunks")


def print_distribution(docs: pd.DataFrame, chunks: pd.DataFrame) -> None:
    print()
    print("CATEGORY DISTRIBUTION")
    print("-" * 100)

    if "category" in docs.columns:
        print("docs category:")
        print(docs["category"].value_counts(dropna=False).to_string())

    if "category" in chunks.columns:
        print()
        print("chunks category:")
        print(chunks["category"].value_counts(dropna=False).to_string())


def main() -> None:
    print("=" * 100)
    print("VALIDATE INDEXES")
    print("=" * 100)
    print(f"ROOT_DIR:              {ROOT_DIR}")
    print(f"DOCS_PATH:             {DOCS_PATH}")
    print(f"CHUNKS_PATH:           {CHUNKS_PATH}")
    print(f"FRIDA_EMBEDDINGS_PATH: {FRIDA_EMBEDDINGS_PATH}")
    print(f"FRIDA_INFO_PATH:       {FRIDA_INFO_PATH}")
    print(f"BM25_PATH:             {BM25_PATH}")

    print()
    print("FILES")
    print("-" * 100)

    require_exists(DOCS_PATH, "docs.parquet")
    require_exists(CHUNKS_PATH, "chunks.parquet")
    require_exists(FRIDA_EMBEDDINGS_PATH, "embeddings.npy")
    require_exists(BM25_PATH, "bm25.pkl")

    print()
    print("LOAD TABLES")
    print("-" * 100)

    try:
        docs = pd.read_parquet(DOCS_PATH)
        ok("docs.parquet loaded")
    except Exception as e:
        fail(f"cannot load docs.parquet: {e}")

    try:
        chunks = pd.read_parquet(CHUNKS_PATH)
        ok("chunks.parquet loaded")
    except Exception as e:
        fail(f"cannot load chunks.parquet: {e}")

    print()
    print("COUNTS")
    print("-" * 100)
    print(f"docs rows:   {len(docs)}")
    print(f"chunks rows: {len(chunks)}")

    validate_docs(docs)
    validate_chunks(chunks, docs)
    validate_embeddings(chunks)
    validate_bm25(chunks)
    validate_gold_dataset(docs, chunks)
    print_distribution(docs, chunks)

    print()
    print("=" * 100)
    ok("indexes validation passed")
    print("=" * 100)


if __name__ == "__main__":
    main()
