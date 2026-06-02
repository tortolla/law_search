from __future__ import annotations

import os
from pathlib import Path


# ======================================================================================
# Project root
# Старое имя ROOT_DIR сохраняем обязательно.
# PROJECT_ROOT добавляем как более явный alias.
# ======================================================================================

ROOT_DIR = Path(
    os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()

PROJECT_ROOT = ROOT_DIR


# ======================================================================================
# Base directories
# Старые имена DATA_DIR, MODELS_DIR, INDEXES_DIR сохраняем.
# ======================================================================================

DATA_DIR = Path(
    os.getenv("DATA_DIR", ROOT_DIR / "data")
).resolve()

MODELS_DIR = Path(
    os.getenv("MODELS_DIR", ROOT_DIR / "models")
).resolve()

RAW_DATA_DIR = Path(
    os.getenv("RAW_DATA_DIR", ROOT_DIR / "data_big")
).resolve()

PROCESSED_DIR = Path(
    os.getenv("PROCESSED_DIR", DATA_DIR / "processed")
).resolve()

INDEXES_DIR = Path(
    os.getenv("INDEXES_DIR", DATA_DIR / "indexes")
).resolve()

# Новый alias, если дальше где-то будет удобнее INDEX_DIR.
INDEX_DIR = INDEXES_DIR


# ======================================================================================
# Yandex data source
# ======================================================================================

YANDEX_PUBLIC_URL = os.getenv(
    "YANDEX_PUBLIC_URL",
    "https://disk.yandex.ru/d/IcLlGxelh0A8GQ",
)

LAW_DIRS = [
    "construction_laws",
    "customs_laws",
    "energy_laws",
    "general_laws",
    "mining_laws",
    "oil_laws",
]


# ======================================================================================
# Processed files
# Старые пути сохраняем.
# ======================================================================================

DOCS_PATH = Path(
    os.getenv("DOCS_PATH", PROCESSED_DIR / "docs.parquet")
).resolve()

CHUNKS_PATH = Path(
    os.getenv("CHUNKS_PATH", PROCESSED_DIR / "chunks.parquet")
).resolve()

CHUNK_STATS_PATH = Path(
    os.getenv("CHUNK_STATS_PATH", PROCESSED_DIR / "chunk_stats.json")
).resolve()

DOCS_JSONL_PATH = Path(
    os.getenv("DOCS_JSONL_PATH", PROCESSED_DIR / "docs.jsonl")
).resolve()

CHUNKS_JSONL_PATH = Path(
    os.getenv("CHUNKS_JSONL_PATH", PROCESSED_DIR / "chunks.jsonl")
).resolve()

BUILD_MANIFEST_PATH = Path(
    os.getenv("BUILD_MANIFEST_PATH", PROCESSED_DIR / "build_manifest.json")
).resolve()


# ======================================================================================
# Index paths
# Старые имена BM25_DIR, FRIDA_DIR, BM25_PATH, FRIDA_EMBEDDINGS_PATH,
# FRIDA_INFO_PATH сохраняем.
# ======================================================================================

BM25_DIR = Path(
    os.getenv("BM25_DIR", INDEXES_DIR / "bm25")
).resolve()

FRIDA_DIR = Path(
    os.getenv("FRIDA_DIR", INDEXES_DIR / "frida")
).resolve()

BM25_PATH = Path(
    os.getenv("BM25_PATH", BM25_DIR / "bm25.pkl")
).resolve()

FRIDA_EMBEDDINGS_PATH = Path(
    os.getenv("FRIDA_EMBEDDINGS_PATH", FRIDA_DIR / "embeddings.npy")
).resolve()

FRIDA_INFO_PATH = Path(
    os.getenv("FRIDA_INFO_PATH", FRIDA_DIR / "model_info.json")
).resolve()

# Новый alias, но старое имя FRIDA_INFO_PATH остаётся главным для совместимости.
FRIDA_MODEL_INFO_PATH = FRIDA_INFO_PATH

FRIDA_MODEL_PATH = Path(
    os.getenv("FRIDA_MODEL_PATH", MODELS_DIR / "FRIDA")
).resolve()


# ======================================================================================
# Chunking parameters
# Эти значения были в старом config.py. Их нельзя терять.
# ======================================================================================

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))
MAX_PARAGRAPH_SIZE = int(os.getenv("MAX_PARAGRAPH_SIZE", "1000"))
LONG_PIECE_OVERLAP = int(os.getenv("LONG_PIECE_OVERLAP", "150"))


# ======================================================================================
# Retrieval defaults
# Важно: сохраняем старые дефолты.
# ======================================================================================

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
DEFAULT_CANDIDATE_K = int(os.getenv("DEFAULT_CANDIDATE_K", "150"))
DEFAULT_BM25_WEIGHT = float(os.getenv("DEFAULT_BM25_WEIGHT", "0.45"))
DEFAULT_FRIDA_WEIGHT = float(os.getenv("DEFAULT_FRIDA_WEIGHT", "0.55"))


# ======================================================================================
# Dense backend
# Пока главный backend — numpy. Milvus добавим позже.
# ======================================================================================

VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "numpy").strip().lower()


# ======================================================================================
# Milvus config
# Пока только настройки. Код Milvus ещё не подключаем.
# ======================================================================================

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "frida_chunks")
MILVUS_METRIC_TYPE = os.getenv("MILVUS_METRIC_TYPE", "IP")
MILVUS_INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "HNSW")


# ======================================================================================
# Eval / reports
# ======================================================================================

GOLD_DATASET_CANDIDATES = [
    PROCESSED_DIR / "dataset_fixed.json",
    ROOT_DIR / "dataset_gold_fixed.json",
    ROOT_DIR / "dataset_fixed.json",
]

EVAL_DIR = Path(
    os.getenv("EVAL_DIR", PROCESSED_DIR)
).resolve()

REPORTS_DIR = Path(
    os.getenv("REPORTS_DIR", ROOT_DIR / "reports")
).resolve()

RESULTS_DIR = Path(
    os.getenv("RESULTS_DIR", ROOT_DIR / "results")
).resolve()


# ======================================================================================
# Helpers
# ======================================================================================

def ensure_dirs() -> None:
    """
    Create standard output directories.
    Safe to call from build scripts.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    BM25_DIR.mkdir(parents=True, exist_ok=True)
    FRIDA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_gold_dataset_path() -> Path | None:
    """
    Return first existing gold dataset path from standard locations.
    """
    for path in GOLD_DATASET_CANDIDATES:
        if path.exists():
            return path
    return None


def print_config() -> None:
    """
    Print resolved config.
    """
    values = {
        "ROOT_DIR": ROOT_DIR,
        "PROJECT_ROOT": PROJECT_ROOT,
        "DATA_DIR": DATA_DIR,
        "MODELS_DIR": MODELS_DIR,
        "RAW_DATA_DIR": RAW_DATA_DIR,
        "PROCESSED_DIR": PROCESSED_DIR,
        "INDEXES_DIR": INDEXES_DIR,
        "DOCS_PATH": DOCS_PATH,
        "CHUNKS_PATH": CHUNKS_PATH,
        "CHUNK_STATS_PATH": CHUNK_STATS_PATH,
        "DOCS_JSONL_PATH": DOCS_JSONL_PATH,
        "CHUNKS_JSONL_PATH": CHUNKS_JSONL_PATH,
        "BM25_DIR": BM25_DIR,
        "BM25_PATH": BM25_PATH,
        "FRIDA_DIR": FRIDA_DIR,
        "FRIDA_EMBEDDINGS_PATH": FRIDA_EMBEDDINGS_PATH,
        "FRIDA_INFO_PATH": FRIDA_INFO_PATH,
        "FRIDA_MODEL_INFO_PATH": FRIDA_MODEL_INFO_PATH,
        "FRIDA_MODEL_PATH": FRIDA_MODEL_PATH,
        "CHUNK_SIZE": CHUNK_SIZE,
        "CHUNK_OVERLAP": CHUNK_OVERLAP,
        "MAX_PARAGRAPH_SIZE": MAX_PARAGRAPH_SIZE,
        "LONG_PIECE_OVERLAP": LONG_PIECE_OVERLAP,
        "DEFAULT_TOP_K": DEFAULT_TOP_K,
        "DEFAULT_CANDIDATE_K": DEFAULT_CANDIDATE_K,
        "DEFAULT_BM25_WEIGHT": DEFAULT_BM25_WEIGHT,
        "DEFAULT_FRIDA_WEIGHT": DEFAULT_FRIDA_WEIGHT,
        "VECTOR_BACKEND": VECTOR_BACKEND,
        "MILVUS_HOST": MILVUS_HOST,
        "MILVUS_PORT": MILVUS_PORT,
        "MILVUS_COLLECTION": MILVUS_COLLECTION,
        "EVAL_DIR": EVAL_DIR,
        "REPORTS_DIR": REPORTS_DIR,
        "RESULTS_DIR": RESULTS_DIR,
    }

    print("=" * 100)
    print("RETRIEVAL CONFIG")
    print("=" * 100)
    for key, value in values.items():
        print(f"{key}: {value}")
    print("=" * 100)
