from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path

from app.retrieval.config import (
    ROOT_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DIR,
    INDEXES_DIR,
    BM25_DIR,
    FRIDA_DIR,
    FRIDA_MODEL_PATH,
    VECTOR_BACKEND,
    MILVUS_HOST,
    MILVUS_PORT,
    MILVUS_COLLECTION,
    ensure_dirs,
    print_config,
)


MIN_PYTHON = (3, 10)


REQUIRED_IMPORTS = [
    "numpy",
    "pandas",
    "tqdm",
    "requests",
    "pyarrow",
    "rank_bm25",
    "sentence_transformers",
]


OPTIONAL_IMPORTS = [
    "pymilvus",
]


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def fail(message: str) -> None:
    print(f"[ERROR] {message}")
    sys.exit(1)


def check_python() -> None:
    print()
    print("PYTHON")
    print("-" * 100)

    version = sys.version_info
    print(f"Python executable: {sys.executable}")
    print(f"Python version:    {platform.python_version()}")

    if (version.major, version.minor) < MIN_PYTHON:
        fail(f"Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]} required")
    ok(f"Python version is compatible: {platform.python_version()}")


def check_project_structure() -> None:
    print()
    print("PROJECT STRUCTURE")
    print("-" * 100)

    required_dirs = [
        ROOT_DIR,
        ROOT_DIR / "app",
        ROOT_DIR / "app" / "retrieval",
        ROOT_DIR / "scripts",
        DATA_DIR,
        ROOT_DIR / "models",
    ]

    for path in required_dirs:
        if not path.exists():
            fail(f"required directory not found: {path}")
        ok(f"directory exists: {path}")

    config_path = ROOT_DIR / "app" / "retrieval" / "config.py"
    if not config_path.exists():
        fail(f"config.py not found: {config_path}")
    ok(f"config.py exists: {config_path}")


def check_writable_dirs() -> None:
    print()
    print("WRITE ACCESS")
    print("-" * 100)

    ensure_dirs()

    writable_dirs = [
        DATA_DIR,
        PROCESSED_DIR,
        INDEXES_DIR,
        BM25_DIR,
        FRIDA_DIR,
    ]

    for directory in writable_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        test_file = directory / ".write_test"

        try:
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink()
            ok(f"writable: {directory}")
        except Exception as e:
            fail(f"not writable: {directory}; error={e}")


def check_imports() -> None:
    print()
    print("PYTHON IMPORTS")
    print("-" * 100)

    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
            ok(f"import works: {module_name}")
        except Exception as e:
            fail(f"cannot import required module '{module_name}': {e}")

    for module_name in OPTIONAL_IMPORTS:
        try:
            importlib.import_module(module_name)
            ok(f"optional import works: {module_name}")
        except Exception as e:
            warn(f"optional module not available: {module_name}; error={e}")


def check_frida_model() -> None:
    print()
    print("FRIDA MODEL")
    print("-" * 100)

    print(f"FRIDA_MODEL_PATH: {FRIDA_MODEL_PATH}")

    if not FRIDA_MODEL_PATH.exists():
        fail(
            "FRIDA model directory not found. "
            f"Expected: {FRIDA_MODEL_PATH}. "
            "Put the model into models/FRIDA or later run scripts/00_download_frida_model.py"
        )

    if not FRIDA_MODEL_PATH.is_dir():
        fail(f"FRIDA_MODEL_PATH exists but is not a directory: {FRIDA_MODEL_PATH}")

    expected_any = [
        "config.json",
        "modules.json",
        "sentence_bert_config.json",
    ]

    found = [name for name in expected_any if (FRIDA_MODEL_PATH / name).exists()]

    if not found:
        warn(
            "FRIDA directory exists, but standard SentenceTransformer files were not found. "
            "This may still work if the model layout is custom."
        )
    else:
        ok(f"FRIDA model files detected: {found}")

    ok("FRIDA model directory check passed")


def check_raw_data_status() -> None:
    print()
    print("RAW DATA STATUS")
    print("-" * 100)

    print(f"RAW_DATA_DIR: {RAW_DATA_DIR}")

    if not RAW_DATA_DIR.exists():
        warn(
            f"raw data directory does not exist yet: {RAW_DATA_DIR}. "
            "This is OK before running scripts/01_download_data.py"
        )
        return

    law_dirs = sorted([p.name for p in RAW_DATA_DIR.iterdir() if p.is_dir() and p.name.endswith("_laws")])
    if law_dirs:
        ok(f"found law directories: {law_dirs}")
    else:
        warn("RAW_DATA_DIR exists, but no *_laws directories found yet")


def check_vector_backend() -> None:
    print()
    print("VECTOR BACKEND")
    print("-" * 100)

    print(f"VECTOR_BACKEND: {VECTOR_BACKEND}")

    if VECTOR_BACKEND not in {"numpy", "milvus"}:
        fail(f"unknown VECTOR_BACKEND: {VECTOR_BACKEND}")

    if VECTOR_BACKEND == "numpy":
        ok("using numpy dense backend")
        return

    if VECTOR_BACKEND == "milvus":
        print(f"MILVUS_HOST: {MILVUS_HOST}")
        print(f"MILVUS_PORT: {MILVUS_PORT}")
        print(f"MILVUS_COLLECTION: {MILVUS_COLLECTION}")

        try:
            from pymilvus import connections

            connections.connect(
                alias="check_env",
                host=MILVUS_HOST,
                port=MILVUS_PORT,
            )
            ok("Milvus connection works")
        except Exception as e:
            fail(f"cannot connect to Milvus at {MILVUS_HOST}:{MILVUS_PORT}; error={e}")


def main() -> None:
    print("=" * 100)
    print("CHECK ENVIRONMENT")
    print("=" * 100)

    print_config()

    check_python()
    check_project_structure()
    check_writable_dirs()
    check_imports()
    check_frida_model()
    check_raw_data_status()
    check_vector_backend()

    print()
    print("=" * 100)
    ok("environment check passed")
    print("=" * 100)


if __name__ == "__main__":
    main()
