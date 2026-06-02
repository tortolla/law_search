from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from app.retrieval.config import (
    CHUNKS_PATH,
    FRIDA_DIR,
    FRIDA_EMBEDDINGS_PATH,
    FRIDA_INFO_PATH,
    FRIDA_MODEL_PATH,
)


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def fail(message: str) -> None:
    print(f"[ERROR] {message}")
    sys.exit(1)


def choose_device(requested: str) -> str:
    requested = requested.lower().strip()

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            fail("device='cuda' requested, but CUDA is not available")
        return "cuda"

    if requested == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            fail("device='mps' requested, but MPS is not available")
        return "mps"

    if requested == "cpu":
        return "cpu"

    fail(f"unknown device: {requested}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FRIDA dense embeddings from chunks.parquet."
    )

    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=CHUNKS_PATH,
        help=f"Path to chunks.parquet. Default: {CHUNKS_PATH}",
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=FRIDA_MODEL_PATH,
        help=f"Path to local FRIDA model. Default: {FRIDA_MODEL_PATH}",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FRIDA_DIR,
        help=f"Output directory. Default: {FRIDA_DIR}",
    )

    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=FRIDA_EMBEDDINGS_PATH,
        help=f"Output .npy path. Default: {FRIDA_EMBEDDINGS_PATH}",
    )

    parser.add_argument(
        "--info-path",
        type=Path,
        default=FRIDA_INFO_PATH,
        help=f"Output model_info.json path. Default: {FRIDA_INFO_PATH}",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device for encoding. Default: auto.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for SentenceTransformer.encode. Default: 64.",
    )

    parser.add_argument(
        "--prompt-name",
        type=str,
        default="search_document",
        help="Prompt name for document embeddings. Default: search_document.",
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Normalize embeddings. Default: True.",
    )

    parser.add_argument(
        "--no-normalize",
        action="store_false",
        dest="normalize",
        help="Disable embedding normalization.",
    )

    parser.add_argument(
        "--save-chunks-copy",
        action="store_true",
        help="Also save a chunks.jsonl copy into output directory.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    chunks_path = args.chunks_path.resolve()
    model_path = args.model_path.resolve()
    out_dir = args.out_dir.resolve()
    embeddings_path = args.embeddings_path.resolve()
    info_path = args.info_path.resolve()

    print("=" * 100)
    print("BUILD FRIDA EMBEDDINGS")
    print("=" * 100)
    print(f"chunks_path:      {chunks_path}")
    print(f"model_path:       {model_path}")
    print(f"out_dir:          {out_dir}")
    print(f"embeddings_path:  {embeddings_path}")
    print(f"info_path:        {info_path}")
    print(f"batch_size:       {args.batch_size}")
    print(f"prompt_name:      {args.prompt_name}")
    print(f"normalize:        {args.normalize}")

    if not chunks_path.exists():
        fail(f"chunks file not found: {chunks_path}")

    if not model_path.exists():
        fail(f"FRIDA model directory not found: {model_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.parent.mkdir(parents=True, exist_ok=True)

    print()
    print("LOAD CHUNKS")
    print("-" * 100)

    chunks_df = pd.read_parquet(chunks_path)
    ok(f"chunks loaded: {len(chunks_df)} rows")

    required_columns = ["chunk_id", "doc_id", "chunk_text"]
    missing = [c for c in required_columns if c not in chunks_df.columns]
    if missing:
        fail(f"chunks file missing required columns: {missing}")

    if not chunks_df["chunk_id"].astype(str).is_unique:
        duplicated = chunks_df["chunk_id"].astype(str).value_counts()
        duplicated = duplicated[duplicated > 1].head(20)
        print(duplicated.to_string())
        fail("chunk_id is not unique")

    chunks_df["chunk_text"] = chunks_df["chunk_text"].fillna("").astype(str)
    empty_chunks = chunks_df["chunk_text"].str.len().eq(0).sum()

    if empty_chunks:
        fail(f"empty chunk_text rows found: {empty_chunks}. Rebuild chunks before embeddings.")

    texts = chunks_df["chunk_text"].tolist()

    print()
    print("DEVICE")
    print("-" * 100)

    device = choose_device(args.device)
    print(f"selected device: {device}")

    if device == "cuda":
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name:  {torch.cuda.get_device_name(0)}")

    if device == "mps":
        warn("MPS selected. If encoding is unstable/slow, use --device cpu or run on CUDA server.")

    print()
    print("LOAD MODEL")
    print("-" * 100)

    model = SentenceTransformer(str(model_path), device=device)
    ok(f"FRIDA loaded from: {model_path}")

    print()
    print("ENCODE")
    print("-" * 100)

    embeddings = model.encode(
        texts,
        prompt_name=args.prompt_name,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=args.normalize,
    )

    embeddings = embeddings.astype("float32", copy=False)

    print(f"embeddings shape: {embeddings.shape}")
    print(f"embeddings dtype: {embeddings.dtype}")

    if embeddings.ndim != 2:
        fail(f"embeddings must be 2D, got shape={embeddings.shape}")

    if embeddings.shape[0] != len(chunks_df):
        fail(f"embeddings rows != chunks rows: {embeddings.shape[0]} != {len(chunks_df)}")

    if args.normalize:
        sample_size = min(1000, len(embeddings))
        norms = np.linalg.norm(embeddings[:sample_size], axis=1)
        print(f"sample norm mean: {float(norms.mean()):.6f}")
        print(f"sample norm min:  {float(norms.min()):.6f}")
        print(f"sample norm max:  {float(norms.max()):.6f}")

        if not np.all(np.isfinite(norms)):
            fail("non-finite embedding norms detected")

    print()
    print("SAVE")
    print("-" * 100)

    np.save(embeddings_path, embeddings)
    ok(f"saved embeddings: {embeddings_path}")

    info = {
        "model_path": str(model_path),
        "chunks_path": str(chunks_path),
        "num_chunks": int(len(chunks_df)),
        "embedding_dim": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
        "prompt_name": args.prompt_name,
        "normalized": bool(args.normalize),
        "device": device,
        "batch_size": int(args.batch_size),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "chunks_format": "parquet",
        "output_embeddings_path": str(embeddings_path),
    }

    info_path.write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    ok(f"saved model info: {info_path}")

    if args.save_chunks_copy:
        chunks_copy_path = out_dir / "chunks.jsonl"
        chunks_df.to_json(
            chunks_copy_path,
            orient="records",
            lines=True,
            force_ascii=False,
        )
        ok(f"saved chunks copy: {chunks_copy_path}")

    print()
    print("=" * 100)
    ok("FRIDA embeddings build completed")
    print("=" * 100)


if __name__ == "__main__":
    main()
