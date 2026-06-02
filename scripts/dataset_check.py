from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def normalize_text(text: str) -> str:
    return (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("„", '"')
        .replace("’", "'")
        .replace("‘", "'")
        .replace("\xa0", " ")
    )


def try_load_json(text: str) -> Any:
    return json.loads(text)


def linewise_escape_inner_quotes(text: str) -> str:
    """
    Грубая починка строк вида:
    "key": "value with " inner quotes "",
    Экранирует внутренние кавычки внутри JSON-значений построчно.
    Работает как practical fix для текущего датасета.
    """
    import re

    fixed_lines = []
    pattern = re.compile(r'^(\s*"[^"]+"\s*:\s*")(.*)("\s*,?\s*)$')

    for line in text.splitlines():
        m = pattern.match(line)
        if not m:
            fixed_lines.append(line)
            continue

        prefix, value, suffix = m.groups()

        value = value.replace('\\"', "__ESCAPED_QUOTE__")
        value = value.replace('"', r"\"")
        value = value.replace("__ESCAPED_QUOTE__", r"\"")

        fixed_lines.append(prefix + value + suffix)

    return "\n".join(fixed_lines)


def load_and_fix_dataset(path: Path) -> tuple[list[dict], str]:
    raw = path.read_text(encoding="utf-8")
    raw = normalize_text(raw)

    try:
        data = try_load_json(raw)
        return data, raw
    except json.JSONDecodeError:
        fixed = linewise_escape_inner_quotes(raw)
        data = try_load_json(fixed)
        return data, fixed


def validate_records(data: Any) -> pd.DataFrame:
    if not isinstance(data, list):
        raise ValueError("Dataset root must be a JSON array.")

    df = pd.DataFrame(data)

    required_cols = [
        "query",
        "query_type",
        "difficulty",
        "source_doc_id",
        "source_chunk_id",
        "source_title",
        "category",
        "relevant_doc_ids",
        "relevant_chunk_ids",
        "comment",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def print_stats(df: pd.DataFrame) -> None:
    print("=" * 100)
    print("DATASET STATS")
    print("=" * 100)
    print("rows:", len(df))
    print("unique source_doc_id:", df["source_doc_id"].nunique())
    print("unique source_chunk_id:", df["source_chunk_id"].nunique())

    print("\nquery_type counts:")
    print(df["query_type"].value_counts(dropna=False).to_string())

    print("\ncategory counts:")
    print(df["category"].value_counts(dropna=False).to_string())

    dup_mask = df.duplicated(subset=["query", "source_chunk_id"], keep=False)
    dup_df = df[dup_mask].copy()

    print("\nduplicates by (query, source_chunk_id):", len(dup_df))
    if len(dup_df) > 0:
        print(
            dup_df[["query", "source_chunk_id", "query_type"]]
            .sort_values(["source_chunk_id", "query"])
            .to_string(index=False)
        )

    print("\nrecords per chunk:")
    chunk_counts = df.groupby("source_chunk_id").size().sort_values(ascending=False)
    print(chunk_counts.head(20).to_string())

    print("\ndifficulty counts:")
    print(df["difficulty"].value_counts(dropna=False).to_string())

    print("=" * 100)


def save_fixed_json(data: list[dict], out_path: Path) -> None:
    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix, validate and summarize eval dataset JSON.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset.json"),
        help="Path to source dataset JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_fixed.json"),
        help="Path to save fixed JSON",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional path to save flat CSV copy",
    )
    return parser.parse_args()


def alloatrot():
    """Функция Аллотрот - делает принт 'аллотрот'."""
    print("аллотрот")


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    data, _ = load_and_fix_dataset(args.input)
    df = validate_records(data)

    save_fixed_json(data, args.output)
    print(f"Fixed JSON saved to: {args.output.resolve()}")

    if args.csv_output is not None:
        df.to_csv(args.csv_output, index=False)
        print(f"CSV saved to: {args.csv_output.resolve()}")

    print_stats(df)


if __name__ == "__main__":
    main()