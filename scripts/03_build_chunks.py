from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm.auto import tqdm

from app.retrieval.config import (
    RAW_DATA_DIR,
    DOCS_PATH,
    CHUNKS_PATH,
    CHUNK_STATS_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_PARAGRAPH_SIZE,
    LONG_PIECE_OVERLAP,
    PROCESSED_DIR,
)

from app.retrieval.utils import clean_text, split_text_into_chunks


DATA_ROOT = RAW_DATA_DIR

ADD_CONTEXT_TO_CHUNK_TEXT = True
MAX_KEYWORDS_IN_CHUNK = 12
MAX_HIERARCHY_CHARS = 500


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="cp1251"))


def normalize_doc_id(value: str) -> str:
    value = str(value).strip()
    value = Path(value).stem
    value = value.replace(".md", "").replace(".txt", "").replace(".pdf", "")
    return value


def read_md_file(md_file: Path) -> str:
    try:
        text = md_file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = md_file.read_text(encoding="cp1251")
        except Exception:
            text = md_file.read_text(errors="ignore")
    return clean_text(text)


def find_md_dirs(category_dir: Path) -> list[Path]:
    return sorted(
        p for p in category_dir.iterdir()
        if p.is_dir() and p.name.endswith("_md")
    )


def load_metadata_map(category_dir: Path) -> tuple[dict[str, dict[str, Any]], str | None]:
    metadata_path = category_dir / "metadata.json"

    if not metadata_path.exists():
        print(f"[WARN] Нет metadata.json: {category_dir}")
        return {}, None

    obj = load_json(metadata_path)

    if not isinstance(obj, dict):
        print(f"[WARN] metadata.json не dict: {metadata_path}")
        return {}, str(metadata_path)

    result: dict[str, dict[str, Any]] = {}

    for raw_doc_id, raw_meta in obj.items():
        doc_id = normalize_doc_id(raw_doc_id)

        if isinstance(raw_meta, dict):
            name = (
                raw_meta.get("name")
                or raw_meta.get("title")
                or raw_meta.get("document_name")
                or raw_meta.get("doc_title")
                or doc_id
            )

            key_words = raw_meta.get("key_words") or raw_meta.get("keywords") or []
            if isinstance(key_words, str):
                key_words = [key_words]
            if not isinstance(key_words, list):
                key_words = []

            result[doc_id] = {
                "name": str(name).strip(),
                "url": raw_meta.get("url"),
                "section_code": raw_meta.get("section_code"),
                "section_title": raw_meta.get("section_title"),
                "downloaded_at": raw_meta.get("downloaded_at"),
                "key_words": [str(x).strip() for x in key_words if str(x).strip()],
            }

        elif isinstance(raw_meta, str):
            result[doc_id] = {
                "name": raw_meta.strip(),
                "url": None,
                "section_code": None,
                "section_title": None,
                "downloaded_at": None,
                "key_words": [],
            }

    print(f"[META] {category_dir.name}: {len(result)} records from metadata.json")
    return result, str(metadata_path)


def flatten_hierarchy(obj: Any, chain: list[str] | None = None) -> list[tuple[str | None, str]]:
    """
    Универсально разворачивает hierarchy.json в пары:
    (doc_id, hierarchy_chain)

    Поддерживает разные структуры:
    - dict с children/items/documents
    - вложенные dict
    - list
    """
    if chain is None:
        chain = []

    rows: list[tuple[str | None, str]] = []

    if isinstance(obj, dict):
        title = (
            obj.get("title")
            or obj.get("name")
            or obj.get("section_title")
            or obj.get("label")
        )

        next_chain = chain[:]
        if title:
            s = str(title).strip()
            if s and s not in next_chain:
                next_chain.append(s)

        doc_id = (
            obj.get("doc_id")
            or obj.get("id")
            or obj.get("document_id")
            or obj.get("eoNumber")
            or obj.get("number")
        )

        if doc_id:
            rows.append((normalize_doc_id(str(doc_id)), " / ".join(next_chain)))

        for key in ("children", "items", "documents", "docs", "data", "records"):
            if key in obj:
                rows.extend(flatten_hierarchy(obj[key], next_chain))

        # На случай структуры {"000...": {...}, "000...": {...}}
        for k, v in obj.items():
            if k in {"children", "items", "documents", "docs", "data", "records"}:
                continue

            if isinstance(v, (dict, list)):
                if str(k).isdigit() or str(k).startswith("000"):
                    sub_rows = flatten_hierarchy(v, next_chain)
                    if sub_rows:
                        rows.extend(sub_rows)
                    else:
                        rows.append((normalize_doc_id(str(k)), " / ".join(next_chain)))
                else:
                    rows.extend(flatten_hierarchy(v, next_chain))

    elif isinstance(obj, list):
        for item in obj:
            rows.extend(flatten_hierarchy(item, chain))

    return rows


def load_hierarchy_map(category_dir: Path) -> tuple[dict[str, str], str | None]:
    hierarchy_path = category_dir / "hierarchy.json"

    if not hierarchy_path.exists():
        return {}, None

    try:
        obj = load_json(hierarchy_path)
    except Exception as e:
        print(f"[WARN] Не удалось прочитать hierarchy.json {hierarchy_path}: {e}")
        return {}, str(hierarchy_path)

    rows = flatten_hierarchy(obj)
    result: dict[str, str] = {}

    for doc_id, chain in rows:
        if doc_id and chain:
            result[doc_id] = chain[:MAX_HIERARCHY_CHARS]

    print(f"[HIER] {category_dir.name}: {len(result)} hierarchy links")
    return result, str(hierarchy_path)


def build_context_prefix(
    *,
    category: str,
    source_group: str,
    title: str,
    section_code: str | None,
    section_title: str | None,
    key_words: list[str],
    hierarchy_chain: str | None,
) -> str:
    lines = []

    lines.append(f"Категория права: {category}")
    lines.append(f"Группа источника: {source_group}")

    if section_title:
        lines.append(f"Раздел источника: {section_title}")

    if section_code:
        lines.append(f"Код раздела: {section_code}")

    if key_words:
        kws = key_words[:MAX_KEYWORDS_IN_CHUNK]
        lines.append("Ключевые слова: " + "; ".join(kws))

    if hierarchy_chain:
        lines.append(f"Иерархия: {hierarchy_chain}")

    if title:
        lines.append(f"Документ: {title}")

    return "\n".join(lines)


def maybe_enrich_chunk_text(context_prefix: str, chunk: str) -> str:
    chunk = clean_text(chunk)

    if not ADD_CONTEXT_TO_CHUNK_TEXT:
        return chunk

    context_prefix = clean_text(context_prefix)

    if not context_prefix:
        return chunk

    return f"{context_prefix}\n\n{chunk}"


def build_docs_df() -> pd.DataFrame:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Не найдена папка: {DATA_ROOT}")

    records = []
    seen_global_ids: set[str] = set()

    category_dirs = sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.endswith("_laws"))

    for category_dir in tqdm(category_dirs, desc="Категории"):
        category = category_dir.name

        metadata_map, meta_file = load_metadata_map(category_dir)
        hierarchy_map, hierarchy_file = load_hierarchy_map(category_dir)

        md_dirs = find_md_dirs(category_dir)
        if not md_dirs:
            print(f"[WARN] Нет *_md папок: {category_dir}")
            continue

        for md_dir in md_dirs:
            source_group = md_dir.name[:-3]
            md_files = sorted(md_dir.rglob("*.md"))

            if not md_files:
                print(f"[WARN] Нет md файлов: {md_dir}")
                continue

            for md_file in tqdm(md_files, desc=f"{category}/{md_dir.name}", leave=False):
                text = read_md_file(md_file)
                if not text:
                    continue

                base_doc_id = normalize_doc_id(md_file.stem)
                doc_id = base_doc_id

                global_key = f"{category}::{doc_id}"
                if global_key in seen_global_ids:
                    doc_id = f"{source_group}__{base_doc_id}"
                    global_key = f"{category}::{doc_id}"

                seen_global_ids.add(global_key)

                meta = metadata_map.get(base_doc_id, {})
                title = meta.get("name") or base_doc_id

                section_code = meta.get("section_code")
                section_title = meta.get("section_title")
                url = meta.get("url")
                downloaded_at = meta.get("downloaded_at")
                key_words = meta.get("key_words") or []

                hierarchy_chain = hierarchy_map.get(base_doc_id)

                context_prefix = build_context_prefix(
                    category=category,
                    source_group=source_group,
                    title=title,
                    section_code=section_code,
                    section_title=section_title,
                    key_words=key_words,
                    hierarchy_chain=hierarchy_chain,
                )

                records.append(
                    {
                        "category": category,
                        "source_group": source_group,
                        "doc_id": doc_id,
                        "base_doc_id": base_doc_id,
                        "title": title,
                        "section_code": section_code,
                        "section_title": section_title,
                        "key_words": key_words,
                        "key_words_text": "; ".join(key_words),
                        "hierarchy_chain": hierarchy_chain,
                        "url": url,
                        "downloaded_at": downloaded_at,
                        "md_path": str(md_file),
                        "txt_path": str(md_file),
                        "meta_file": meta_file,
                        "hierarchy_file": hierarchy_file,
                        "context_prefix": context_prefix,
                        "text": text,
                        "doc_len": len(text),
                    }
                )

    return pd.DataFrame(records)


def build_chunks_df(docs_df: pd.DataFrame) -> pd.DataFrame:
    chunk_records = []

    for row in tqdm(
        docs_df.itertuples(index=False),
        total=len(docs_df),
        desc="Режем на чанки",
    ):
        chunks = split_text_into_chunks(
            row.text,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            max_paragraph_size=MAX_PARAGRAPH_SIZE,
            long_piece_overlap=LONG_PIECE_OVERLAP,
        )

        for i, chunk in enumerate(chunks):
            chunk_text = maybe_enrich_chunk_text(row.context_prefix, chunk)

            chunk_records.append(
                {
                    "doc_id": row.doc_id,
                    "base_doc_id": row.base_doc_id,
                    "title": row.title,
                    "category": row.category,
                    "source_group": row.source_group,
                    "section_code": row.section_code,
                    "section_title": row.section_title,
                    "key_words": row.key_words,
                    "key_words_text": row.key_words_text,
                    "hierarchy_chain": row.hierarchy_chain,
                    "url": row.url,
                    "downloaded_at": row.downloaded_at,
                    "md_path": row.md_path,
                    "txt_path": row.md_path,
                    "meta_file": row.meta_file,
                    "hierarchy_file": row.hierarchy_file,
                    "chunk_id": f"{row.category}__{row.doc_id}__{i}",
                    "chunk_ix": i,
                    "chunk_text": chunk_text,
                    "chunk_len": len(chunk_text),
                }
            )

    return pd.DataFrame(chunk_records)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    docs_df = build_docs_df()

    if docs_df.empty:
        raise RuntimeError("Не найдено ни одного md-документа. Проверь data_big.")

    chunks_df = build_chunks_df(docs_df)

    docs_df.to_parquet(DOCS_PATH, index=False)
    chunks_df.to_parquet(CHUNKS_PATH, index=False)

    docs_jsonl_path = PROCESSED_DIR / "docs.jsonl"
    chunks_jsonl_path = PROCESSED_DIR / "chunks.jsonl"

    docs_df.to_json(docs_jsonl_path, orient="records", lines=True, force_ascii=False)
    chunks_df.to_json(chunks_jsonl_path, orient="records", lines=True, force_ascii=False)

    stats = {
        "data_root": str(DATA_ROOT),
        "num_categories": int(docs_df["category"].nunique()),
        "num_source_groups": int(docs_df[["category", "source_group"]].drop_duplicates().shape[0]),
        "num_docs": int(len(docs_df)),
        "num_chunks": int(len(chunks_df)),
        "docs_with_keywords": int(docs_df["key_words_text"].astype(bool).sum()),
        "docs_with_hierarchy": int(docs_df["hierarchy_chain"].astype(bool).sum()),
        "avg_doc_len": float(docs_df["doc_len"].mean()) if len(docs_df) else 0.0,
        "avg_chunk_len": float(chunks_df["chunk_len"].mean()) if len(chunks_df) else 0.0,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "max_paragraph_size": MAX_PARAGRAPH_SIZE,
        "long_piece_overlap": LONG_PIECE_OVERLAP,
        "add_context_to_chunk_text": ADD_CONTEXT_TO_CHUNK_TEXT,
        "max_keywords_in_chunk": MAX_KEYWORDS_IN_CHUNK,
        "max_hierarchy_chars": MAX_HIERARCHY_CHARS,
    }

    with open(CHUNK_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[OK] docs parquet   -> {DOCS_PATH}")
    print(f"[OK] chunks parquet -> {CHUNKS_PATH}")
    print(f"[OK] docs jsonl     -> {docs_jsonl_path}")
    print(f"[OK] chunks jsonl   -> {chunks_jsonl_path}")
    print(f"[OK] stats          -> {CHUNK_STATS_PATH}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
