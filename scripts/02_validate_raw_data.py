from __future__ import annotations

import json
import sys
from pathlib import Path

from app.retrieval.config import RAW_DATA_DIR


ROOT_LAWS_SUFFIX = "_laws"
MD_DIR_SUFFIX = "_md"


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def fail(message: str) -> None:
    print(f"[ERROR] {message}")
    sys.exit(1)


def load_json(path: Path) -> object:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        fail(f"cannot read json: {path}; error={e}")


def validate_laws_dir(laws_dir: Path) -> dict[str, int]:
    print()
    print("=" * 100)
    print(f"VALIDATE LAWS DIR: {laws_dir}")
    print("=" * 100)

    if not laws_dir.is_dir():
        fail(f"not a directory: {laws_dir}")

    metadata_path = laws_dir / "metadata.json"
    hierarchy_path = laws_dir / "hierarchy.json"

    if not metadata_path.exists():
        fail(f"metadata.json not found: {metadata_path}")
    ok(f"metadata.json exists: {metadata_path}")

    if not hierarchy_path.exists():
        fail(f"hierarchy.json not found: {hierarchy_path}")
    ok(f"hierarchy.json exists: {hierarchy_path}")

    metadata = load_json(metadata_path)
    hierarchy = load_json(hierarchy_path)

    if isinstance(metadata, dict):
        ok(f"metadata.json loaded; records: {len(metadata)}")
    else:
        warn(f"metadata.json is not dict: {type(metadata)}")

    if isinstance(hierarchy, dict):
        ok(f"hierarchy.json loaded; top-level keys: {len(hierarchy)}")
    else:
        warn(f"hierarchy.json is not dict: {type(hierarchy)}")

    md_dirs = sorted([p for p in laws_dir.iterdir() if p.is_dir() and p.name.endswith(MD_DIR_SUFFIX)])
    non_md_dirs = sorted([p for p in laws_dir.iterdir() if p.is_dir() and not p.name.endswith(MD_DIR_SUFFIX)])

    if not md_dirs:
        fail(f"no *_md directories found inside: {laws_dir}")

    ok(f"*_md directories found: {len(md_dirs)}")
    for p in md_dirs[:30]:
        print(f"  + {p.relative_to(RAW_DATA_DIR)}")
    if len(md_dirs) > 30:
        print(f"  ... {len(md_dirs) - 30} more")

    if non_md_dirs:
        warn(f"non-md directories exist inside {laws_dir.name}; pipeline will ignore them")
        for p in non_md_dirs[:30]:
            print(f"  - {p.relative_to(RAW_DATA_DIR)}")
        if len(non_md_dirs) > 30:
            print(f"  ... {len(non_md_dirs) - 30} more")
    else:
        ok("no non-md directories inside laws dir")

    md_files = []
    for md_dir in md_dirs:
        md_files.extend(md_dir.rglob("*.md"))

    if not md_files:
        fail(f"no .md files found in *_md dirs inside: {laws_dir}")

    ok(f"markdown files found: {len(md_files)}")

    empty_md_files = [p for p in md_files if p.stat().st_size == 0]
    if empty_md_files:
        warn(f"empty markdown files: {len(empty_md_files)}")
        for p in empty_md_files[:20]:
            print(f"  - {p.relative_to(RAW_DATA_DIR)}")
    else:
        ok("no empty markdown files")

    md_doc_ids = {p.stem for p in md_files}

    if isinstance(metadata, dict):
        metadata_doc_ids = set(str(k) for k in metadata.keys())
        md_without_meta = sorted(md_doc_ids - metadata_doc_ids)
        meta_without_md = sorted(metadata_doc_ids - md_doc_ids)

        if md_without_meta:
            warn(f"md files without metadata records: {len(md_without_meta)}")
            print("examples:", md_without_meta[:20])
        else:
            ok("all md files have metadata records")

        if meta_without_md:
            warn(f"metadata records without md files: {len(meta_without_md)}")
            print("examples:", meta_without_md[:20])
        else:
            ok("all metadata records have md files")

    return {
        "md_dirs": len(md_dirs),
        "non_md_dirs": len(non_md_dirs),
        "md_files": len(md_files),
        "empty_md_files": len(empty_md_files),
    }


def main() -> None:
    print("=" * 100)
    print("VALIDATE RAW DATA")
    print("=" * 100)
    print(f"RAW_DATA_DIR: {RAW_DATA_DIR}")

    if not RAW_DATA_DIR.exists():
        fail(f"RAW_DATA_DIR does not exist: {RAW_DATA_DIR}")

    if not RAW_DATA_DIR.is_dir():
        fail(f"RAW_DATA_DIR is not a directory: {RAW_DATA_DIR}")

    root_dirs = sorted([p for p in RAW_DATA_DIR.iterdir() if p.is_dir()])

    if not root_dirs:
        fail(f"RAW_DATA_DIR has no subdirectories: {RAW_DATA_DIR}")

    laws_dirs = [p for p in root_dirs if p.name.endswith(ROOT_LAWS_SUFFIX)]
    non_laws_dirs = [p for p in root_dirs if not p.name.endswith(ROOT_LAWS_SUFFIX)]

    if not laws_dirs:
        fail(f"no top-level directories ending with {ROOT_LAWS_SUFFIX!r} found in {RAW_DATA_DIR}")

    print()
    print("TOP-LEVEL *_laws DIRS")
    print("-" * 100)
    for p in laws_dirs:
        print(f"  + {p.name}")

    if non_laws_dirs:
        print()
        print("TOP-LEVEL NON-LAWS DIRS")
        print("-" * 100)
        warn("non-laws dirs exist; pipeline should ignore them")
        for p in non_laws_dirs:
            print(f"  - {p.name}")
    else:
        ok("no non-laws top-level dirs")

    total = {
        "laws_dirs": len(laws_dirs),
        "md_dirs": 0,
        "non_md_dirs": 0,
        "md_files": 0,
        "empty_md_files": 0,
    }

    for laws_dir in laws_dirs:
        stats = validate_laws_dir(laws_dir)
        for key, value in stats.items():
            total[key] += value

    print()
    print("=" * 100)
    print("RAW DATA SUMMARY")
    print("=" * 100)
    for key, value in total.items():
        print(f"{key}: {value}")

    print()
    ok("raw data validation passed")


if __name__ == "__main__":
    main()
