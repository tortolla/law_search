from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from app.retrieval.config import RAW_DATA_DIR, YANDEX_PUBLIC_URL


API_URL = "https://cloud-api.yandex.net/v1/disk/public/resources"
ROOT_LAWS_SUFFIX = "_laws"


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def fail(message: str) -> None:
    print(f"[ERROR] {message}")
    sys.exit(1)


def get_resource(path: str = "", offset: int = 0, limit: int = 1000) -> dict[str, Any]:
    params = {
        "public_key": YANDEX_PUBLIC_URL,
        "path": path,
        "limit": limit,
        "offset": offset,
    }

    try:
        response = requests.get(API_URL, params=params, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        fail(f"cannot get Yandex resource path={path!r}: {e}")


def get_all_items(path: str = "") -> list[dict[str, Any]]:
    all_items: list[dict[str, Any]] = []
    offset = 0
    limit = 1000

    while True:
        resource = get_resource(path=path, offset=offset, limit=limit)
        embedded = resource.get("_embedded", {})
        items = embedded.get("items", [])

        all_items.extend(items)

        total = embedded.get("total", len(all_items))
        if len(all_items) >= total or not items:
            break

        offset += limit

    return all_items


def rel_path(yandex_path: str) -> Path:
    clean = yandex_path.replace("disk:", "").lstrip("/")
    return Path(clean)


def download_file(yandex_file_path: str, out_dir: Path, dry_run: bool) -> None:
    out_path = out_dir / rel_path(yandex_file_path)

    if dry_run:
        print(f"[DRY-RUN] would download file: {yandex_file_path} -> {out_path}")
        return

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[SKIP] exists: {out_path}")
        return

    params = {
        "public_key": YANDEX_PUBLIC_URL,
        "path": yandex_file_path,
    }

    try:
        meta = requests.get(API_URL, params=params, timeout=60)
        meta.raise_for_status()
        download_url = meta.json().get("file")
    except Exception as e:
        fail(f"cannot get download url for {yandex_file_path}: {e}")

    if not download_url:
        warn(f"no download url for: {yandex_file_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(download_url, stream=True, timeout=120) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))

            with open(out_path, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=str(out_path),
            ) as pbar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    except Exception as e:
        if out_path.exists():
            out_path.unlink()
        fail(f"download failed for {yandex_file_path}: {e}")


def download_dir_recursive(yandex_dir_path: str, out_dir: Path, dry_run: bool, counters: dict[str, int]) -> None:
    for item in get_all_items(yandex_dir_path):
        item_type = item.get("type")
        item_path = item.get("path")
        item_name = item.get("name")

        if not item_path:
            warn(f"item without path in {yandex_dir_path}: {item}")
            continue

        if item_type == "file":
            counters["files_selected"] += 1
            download_file(item_path, out_dir=out_dir, dry_run=dry_run)

        elif item_type == "dir":
            counters["dirs_selected"] += 1
            print(f"[DIR] recurse: {item_path}")
            download_dir_recursive(item_path, out_dir=out_dir, dry_run=dry_run, counters=counters)

        else:
            warn(f"unknown item type={item_type} name={item_name}")


def scan_laws_dir(yandex_laws_path: str, out_dir: Path, dry_run: bool, counters: dict[str, int]) -> None:
    """
    Inside one *_laws directory:
    - select .json files;
    - select *_md directories recursively;
    - skip non-md directories.
    """
    print()
    print("=" * 100)
    print(f"SCAN LAWS DIR: {yandex_laws_path}")
    print("=" * 100)

    for item in get_all_items(yandex_laws_path):
        item_type = item.get("type")
        item_name = item.get("name", "")
        item_path = item.get("path")

        if not item_path:
            warn(f"item without path in {yandex_laws_path}: {item}")
            continue

        if item_type == "file":
            if item_name.lower().endswith(".json"):
                counters["files_selected"] += 1
                download_file(item_path, out_dir=out_dir, dry_run=dry_run)
            else:
                counters["files_skipped"] += 1
                print(f"[SKIP] non-json file: {item_path}")

        elif item_type == "dir":
            if item_name.endswith("_md"):
                counters["dirs_selected"] += 1
                print(f"[MD DIR] select recursively: {item_path}")
                download_dir_recursive(item_path, out_dir=out_dir, dry_run=dry_run, counters=counters)
            else:
                counters["dirs_skipped"] += 1
                print(f"[SKIP] non-md dir inside laws dir: {item_path}")

        else:
            warn(f"unknown item type={item_type} name={item_name}")


def scan_root(out_dir: Path, dry_run: bool) -> None:
    """
    Root rule:
    - select every top-level directory ending with *_laws;
    - skip all other top-level directories.
    """
    counters = {
        "root_laws_dirs": 0,
        "root_dirs_skipped": 0,
        "dirs_selected": 0,
        "dirs_skipped": 0,
        "files_selected": 0,
        "files_skipped": 0,
    }

    print("=" * 100)
    print("DOWNLOAD DATA FROM YANDEX DISK")
    print("=" * 100)
    print(f"YANDEX_PUBLIC_URL: {YANDEX_PUBLIC_URL}")
    print(f"OUT_DIR:           {out_dir}")
    print(f"ROOT_LAWS_SUFFIX:  {ROOT_LAWS_SUFFIX}")
    print(f"DRY_RUN:           {dry_run}")

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    root_items = get_all_items("")
    root_dirs = [item for item in root_items if item.get("type") == "dir"]

    available_names = sorted([item.get("name", "") for item in root_dirs])
    print()
    print("Available root dirs:")
    for name in available_names:
        print(f"  - {name}")

    laws_dirs = []
    skipped_dirs = []

    for item in root_dirs:
        item_name = item.get("name", "")

        if item_name.endswith(ROOT_LAWS_SUFFIX):
            laws_dirs.append(item)
        else:
            skipped_dirs.append(item)

    if not laws_dirs:
        fail(f"no top-level directories ending with {ROOT_LAWS_SUFFIX!r} found")

    counters["root_laws_dirs"] = len(laws_dirs)
    counters["root_dirs_skipped"] = len(skipped_dirs)

    print()
    print("Selected top-level *_laws dirs:")
    for item in laws_dirs:
        print(f"  + {item.get('name')}")

    print()
    print("Skipped top-level dirs:")
    for item in skipped_dirs:
        print(f"  - {item.get('name')}")

    for item in laws_dirs:
        scan_laws_dir(item["path"], out_dir=out_dir, dry_run=dry_run, counters=counters)

    print()
    print("=" * 100)
    print("DOWNLOAD SELECTION SUMMARY")
    print("=" * 100)
    for key, value in counters.items():
        print(f"{key}: {value}")

    if dry_run:
        ok("dry-run completed; no files were written")
    else:
        ok("download completed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download only top-level *_laws folders from Yandex Disk; inside them only .json and *_md folders."
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help=f"Output directory. Default: {RAW_DATA_DIR}",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not download files. Only print what would be selected.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scan_root(out_dir=args.out_dir.resolve(), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
