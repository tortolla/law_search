from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from app.retrieval.config import (
    CHUNKS_PATH,
    FRIDA_EMBEDDINGS_PATH,
    MILVUS_COLLECTION,
    MILVUS_HOST,
    MILVUS_PORT,
)


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def fail(message: str) -> None:
    print(f"[ERROR] {message}")
    sys.exit(1)


def safe_str(value: Any, max_len: int) -> str:
    if value is None:
        return ""
    s = str(value)
    if len(s) > max_len:
        return s[:max_len]
    return s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load FRIDA embeddings and chunk metadata into Milvus."
    )

    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=CHUNKS_PATH,
        help=f"Path to chunks.parquet. Default: {CHUNKS_PATH}",
    )

    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=FRIDA_EMBEDDINGS_PATH,
        help=f"Path to embeddings.npy. Default: {FRIDA_EMBEDDINGS_PATH}",
    )

    parser.add_argument(
        "--collection",
        type=str,
        default=MILVUS_COLLECTION,
        help=f"Milvus collection name. Default: {MILVUS_COLLECTION}",
    )

    parser.add_argument(
        "--host",
        type=str,
        default=MILVUS_HOST,
        help=f"Milvus host. Default: {MILVUS_HOST}",
    )

    parser.add_argument(
        "--port",
        type=str,
        default=MILVUS_PORT,
        help=f"Milvus port. Default: {MILVUS_PORT}",
    )

    parser.add_argument(
        "--mode",
        choices=["server", "lite"],
        default="server",
        help="server = host/port Milvus; lite = local Milvus Lite file.",
    )

    parser.add_argument(
        "--uri",
        type=str,
        default="data/milvus_lite/frida_chunks.db",
        help="Milvus Lite URI/file path. Used only with --mode lite.",
    )

    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing collection before loading.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Insert batch size. Default: 5000.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for debug loading.",
    )

    parser.add_argument(
        "--metric-type",
        choices=["IP", "COSINE", "L2"],
        default="IP",
        help="Vector metric. Default: IP. Use IP for normalized FRIDA vectors.",
    )

    parser.add_argument(
        "--index-type",
        choices=["HNSW", "IVF_FLAT", "AUTOINDEX"],
        default="HNSW",
        help="Milvus vector index type. Default: HNSW.",
    )

    return parser.parse_args()


def connect(args: argparse.Namespace) -> None:
    print()
    print("CONNECT MILVUS")
    print("-" * 100)

    if args.mode == "server":
        print(f"host: {args.host}")
        print(f"port: {args.port}")
        connections.connect(alias="default", host=args.host, port=args.port)
        ok("connected to Milvus server")
        return

    uri_path = Path(args.uri)
    uri_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"uri: {uri_path}")
    connections.connect(alias="default", uri=str(uri_path))
    ok("connected to Milvus Lite")


def load_inputs(chunks_path: Path, embeddings_path: Path, limit: int | None) -> tuple[pd.DataFrame, np.ndarray]:
    print()
    print("LOAD INPUTS")
    print("-" * 100)

    chunks_path = chunks_path.resolve()
    embeddings_path = embeddings_path.resolve()

    print(f"chunks_path:     {chunks_path}")
    print(f"embeddings_path: {embeddings_path}")

    if not chunks_path.exists():
        fail(f"chunks.parquet not found: {chunks_path}")

    if not embeddings_path.exists():
        fail(f"embeddings.npy not found: {embeddings_path}")

    chunks = pd.read_parquet(chunks_path)
    embeddings = np.load(embeddings_path)

    print(f"chunks rows:      {len(chunks)}")
    print(f"embeddings shape: {embeddings.shape}")
    print(f"embeddings dtype:  {embeddings.dtype}")

    required_cols = [
        "chunk_id",
        "doc_id",
        "category",
        "source_group",
        "title",
        "chunk_ix",
    ]
    missing = [c for c in required_cols if c not in chunks.columns]
    if missing:
        fail(f"chunks.parquet missing required columns: {missing}")

    if embeddings.ndim != 2:
        fail(f"embeddings must be 2D, got shape={embeddings.shape}")

    if len(chunks) != embeddings.shape[0]:
        fail(f"chunks rows != embeddings rows: {len(chunks)} != {embeddings.shape[0]}")

    if not chunks["chunk_id"].astype(str).is_unique:
        fail("chunk_id is not unique")

    embeddings = embeddings.astype("float32", copy=False)

    if limit is not None:
        if limit <= 0:
            fail("--limit must be positive")
        chunks = chunks.iloc[:limit].copy()
        embeddings = embeddings[:limit]
        warn(f"debug limit applied: {limit}")

    ok("inputs loaded and validated")

    return chunks, embeddings


def make_schema(dim: int) -> CollectionSchema:
    fields = [
        FieldSchema(
            name="pk",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
            description="Auto-generated primary key.",
        ),
        FieldSchema(
            name="chunk_id",
            dtype=DataType.VARCHAR,
            max_length=256,
            description="Unique chunk id.",
        ),
        FieldSchema(
            name="doc_id",
            dtype=DataType.VARCHAR,
            max_length=128,
            description="Document id.",
        ),
        FieldSchema(
            name="category",
            dtype=DataType.VARCHAR,
            max_length=128,
            description="Top-level legal category.",
        ),
        FieldSchema(
            name="source_group",
            dtype=DataType.VARCHAR,
            max_length=128,
            description="Source group.",
        ),
        FieldSchema(
            name="title",
            dtype=DataType.VARCHAR,
            max_length=2048,
            description="Document title.",
        ),
        FieldSchema(
            name="chunk_ix",
            dtype=DataType.INT64,
            description="Chunk index inside document.",
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=dim,
            description="FRIDA normalized dense embedding.",
        ),
    ]

    return CollectionSchema(
        fields=fields,
        description="FRIDA chunk embeddings for legal retrieval.",
        enable_dynamic_field=False,
    )


def create_or_reset_collection(
    collection_name: str,
    dim: int,
    drop_existing: bool,
) -> Collection:
    print()
    print("CREATE COLLECTION")
    print("-" * 100)

    print(f"collection:    {collection_name}")
    print(f"embedding dim: {dim}")
    print(f"drop_existing: {drop_existing}")

    existing = utility.has_collection(collection_name)

    if existing and drop_existing:
        utility.drop_collection(collection_name)
        ok(f"dropped existing collection: {collection_name}")
        existing = False

    if existing:
        collection = Collection(collection_name)
        ok(f"using existing collection: {collection_name}")

        existing_dim = None
        for field in collection.schema.fields:
            if field.name == "embedding":
                existing_dim = field.params.get("dim")

        if int(existing_dim) != int(dim):
            fail(f"existing collection dim != embeddings dim: {existing_dim} != {dim}")

        return collection

    schema = make_schema(dim)
    collection = Collection(name=collection_name, schema=schema)
    ok(f"created collection: {collection_name}")

    return collection


def make_index_params(index_type: str, metric_type: str) -> dict[str, Any]:
    if index_type == "HNSW":
        return {
            "index_type": "HNSW",
            "metric_type": metric_type,
            "params": {
                "M": 16,
                "efConstruction": 200,
            },
        }

    if index_type == "IVF_FLAT":
        return {
            "index_type": "IVF_FLAT",
            "metric_type": metric_type,
            "params": {
                "nlist": 1024,
            },
        }

    if index_type == "AUTOINDEX":
        return {
            "index_type": "AUTOINDEX",
            "metric_type": metric_type,
            "params": {},
        }

    fail(f"unknown index type: {index_type}")


def insert_batches(
    collection: Collection,
    chunks: pd.DataFrame,
    embeddings: np.ndarray,
    batch_size: int,
) -> None:
    print()
    print("INSERT DATA")
    print("-" * 100)

    total = len(chunks)
    print(f"rows to insert: {total}")
    print(f"batch_size:     {batch_size}")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        part = chunks.iloc[start:end]

        entities = [
            part["chunk_id"].map(lambda x: safe_str(x, 256)).tolist(),
            part["doc_id"].map(lambda x: safe_str(x, 128)).tolist(),
            part["category"].map(lambda x: safe_str(x, 128)).tolist(),
            part["source_group"].map(lambda x: safe_str(x, 128)).tolist(),
            part["title"].map(lambda x: safe_str(x, 2048)).tolist(),
            part["chunk_ix"].fillna(0).astype(int).tolist(),
            embeddings[start:end].tolist(),
        ]

        collection.insert(entities)

        print(f"[INSERT] {end}/{total}")

    print("[FLUSH] start")
    collection.flush(timeout=120)
    print("[FLUSH] done")
    ok("insert completed and flushed")


def create_index_and_load(
    collection: Collection,
    index_type: str,
    metric_type: str,
) -> None:
    print()
    print("CREATE INDEX")
    print("-" * 100)

    index_params = make_index_params(index_type=index_type, metric_type=metric_type)
    print(f"index_params: {index_params}")

    existing_indexes = collection.indexes
    if existing_indexes:
        warn(f"collection already has indexes: {existing_indexes}")
    else:
        collection.create_index(field_name="embedding", index_params=index_params)
        ok("vector index created")

    collection.load()
    ok("collection loaded into memory")


def validate_collection(collection: Collection, expected_count: int) -> None:
    print()
    print("VALIDATE COLLECTION")
    print("-" * 100)

    print("[FLUSH] validation start")
    collection.flush(timeout=120)
    print("[FLUSH] validation done")
    actual_count = collection.num_entities

    print(f"expected_count: {expected_count}")
    print(f"actual_count:   {actual_count}")

    if int(actual_count) != int(expected_count):
        fail(f"Milvus num_entities != expected count: {actual_count} != {expected_count}")

    ok("Milvus entity count matches expected count")

    print()
    print("SAMPLE QUERY")
    print("-" * 100)

    results = collection.query(
        expr="chunk_ix >= 0",
        output_fields=["chunk_id", "doc_id", "category", "source_group", "title", "chunk_ix"],
        limit=3,
    )

    for i, row in enumerate(results, start=1):
        print(f"[{i}] {row}")

    ok("sample query works")


def main() -> None:
    args = parse_args()

    print("=" * 100)
    print("LOAD FRIDA EMBEDDINGS INTO MILVUS")
    print("=" * 100)
    print(f"created_at:      {datetime.now().isoformat(timespec='seconds')}")
    print(f"mode:            {args.mode}")
    print(f"collection:      {args.collection}")
    print(f"metric_type:     {args.metric_type}")
    print(f"index_type:      {args.index_type}")

    connect(args)

    chunks, embeddings = load_inputs(
        chunks_path=args.chunks_path,
        embeddings_path=args.embeddings_path,
        limit=args.limit,
    )

    collection = create_or_reset_collection(
        collection_name=args.collection,
        dim=embeddings.shape[1],
        drop_existing=args.drop_existing,
    )

    if collection.num_entities > 0 and not args.drop_existing:
        warn(
            f"collection already contains {collection.num_entities} entities. "
            "Use --drop-existing to rebuild it from scratch."
        )
        validate_collection(collection, expected_count=collection.num_entities)
        return

    insert_batches(
        collection=collection,
        chunks=chunks,
        embeddings=embeddings,
        batch_size=args.batch_size,
    )

    create_index_and_load(
        collection=collection,
        index_type=args.index_type,
        metric_type=args.metric_type,
    )

    validate_collection(
        collection=collection,
        expected_count=len(chunks),
    )

    print()
    print("=" * 100)
    ok("Milvus load completed")
    print("=" * 100)


if __name__ == "__main__":
    main()
