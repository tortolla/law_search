from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = [
    "id",
    "question",
    "question_type",
    "difficulty",
    "expected_behavior",
    "expected_answer",
    "gold_points",
    "source_doc_id",
    "source_chunk_id",
    "source_title",
    "category",
    "relevant_doc_ids",
    "relevant_chunk_ids",
    "must_cite",
    "must_not_claim",
    "comment",
]

ALLOWED_QUESTION_TYPES = {
    "direct_legal_fact",
    "procedure_or_condition",
    "power_or_authority",
    "definition_or_scope",
    "list_extraction",
    "application_to_situation",
    "negative_or_insufficient_basis",
}

ALLOWED_DIFFICULTY = {"easy", "medium", "hard"}
ALLOWED_EXPECTED_BEHAVIOR = {"answer", "insufficient_basis"}


def load_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"Empty JSON file: {path}")
    return json.loads(text)


def save_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def as_list_dataset(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        # На случай, если случайно сохранился {"items": [...]}.
        for key in ("items", "data", "examples", "questions"):
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError("Dataset must be a JSON array or object with items/data/examples/questions list")


def validate_item(item: dict[str, Any], ix: int) -> list[str]:
    errors: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in item:
            errors.append(f"[{ix}] missing field: {field}")

    for field in REQUIRED_FIELDS:
        if field not in item:
            continue

        value = item[field]

        if field in {
            "id",
            "question",
            "question_type",
            "difficulty",
            "expected_behavior",
            "expected_answer",
            "source_doc_id",
            "source_chunk_id",
            "source_title",
            "category",
            "comment",
        }:
            if not isinstance(value, str):
                errors.append(f"[{ix}] field must be string: {field}")
            elif field != "id" and not value.strip():
                errors.append(f"[{ix}] field is empty: {field}")

        if field in {
            "gold_points",
            "relevant_doc_ids",
            "relevant_chunk_ids",
            "must_cite",
            "must_not_claim",
        }:
            if not isinstance(value, list):
                errors.append(f"[{ix}] field must be list: {field}")
            elif field in {"gold_points", "relevant_doc_ids", "relevant_chunk_ids"} and len(value) == 0:
                errors.append(f"[{ix}] field list is empty: {field}")
            else:
                for j, x in enumerate(value):
                    if not isinstance(x, str):
                        errors.append(f"[{ix}] {field}[{j}] must be string")
                    elif field in {"gold_points", "relevant_doc_ids", "relevant_chunk_ids"} and not x.strip():
                        errors.append(f"[{ix}] {field}[{j}] is empty")

    qt = item.get("question_type")
    if isinstance(qt, str) and qt not in ALLOWED_QUESTION_TYPES:
        errors.append(f"[{ix}] unknown question_type: {qt}")

    difficulty = item.get("difficulty")
    if isinstance(difficulty, str) and difficulty not in ALLOWED_DIFFICULTY:
        errors.append(f"[{ix}] unknown difficulty: {difficulty}")

    behavior = item.get("expected_behavior")
    if isinstance(behavior, str) and behavior not in ALLOWED_EXPECTED_BEHAVIOR:
        errors.append(f"[{ix}] unknown expected_behavior: {behavior}")

    source_doc_id = item.get("source_doc_id")
    relevant_doc_ids = item.get("relevant_doc_ids")
    if isinstance(source_doc_id, str) and isinstance(relevant_doc_ids, list):
        if source_doc_id not in relevant_doc_ids:
            errors.append(f"[{ix}] source_doc_id not in relevant_doc_ids: {source_doc_id}")

    source_chunk_id = item.get("source_chunk_id")
    relevant_chunk_ids = item.get("relevant_chunk_ids")
    if isinstance(source_chunk_id, str) and isinstance(relevant_chunk_ids, list):
        if source_chunk_id not in relevant_chunk_ids:
            errors.append(f"[{ix}] source_chunk_id not in relevant_chunk_ids: {source_chunk_id}")

    if item.get("expected_behavior") == "insufficient_basis":
        if item.get("question_type") != "negative_or_insufficient_basis":
            errors.append(f"[{ix}] insufficient_basis should usually have question_type=negative_or_insufficient_basis")

    return errors


def print_counter(title: str, counter: Counter) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
    if not counter:
        print("(empty)")
        return

    for key, value in counter.most_common():
        print(f"{key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate, assign ids, and summarize legal QA gold dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset_gold.json"),
        help="Path to dataset_gold.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_gold_fixed.json"),
        help="Path to write fixed dataset",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="qa",
        help="ID prefix, e.g. qa -> qa001",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input file instead of writing --output",
    )
    args = parser.parse_args()

    data = load_json(args.input)
    items = as_list_dataset(data)

    if not all(isinstance(x, dict) for x in items):
        raise ValueError("All dataset items must be JSON objects")

    # Нормализуем id заново, чтобы не было пустых/дублей/съехавших номеров.
    width = max(3, len(str(len(items))))
    for i, item in enumerate(items, start=1):
        item["id"] = f"{args.prefix}{i:0{width}d}"

    all_errors: list[str] = []
    for ix, item in enumerate(items, start=1):
        all_errors.extend(validate_item(item, ix))

    ids = [item.get("id") for item in items]
    duplicate_ids = [k for k, v in Counter(ids).items() if v > 1]
    if duplicate_ids:
        all_errors.append(f"duplicate ids: {duplicate_ids}")

    questions = [item.get("question") for item in items]
    duplicate_questions = [k for k, v in Counter(questions).items() if isinstance(k, str) and v > 1]
    if duplicate_questions:
        all_errors.append(f"duplicate questions: {duplicate_questions}")

    out_path = args.input if args.in_place else args.output
    save_json(out_path, items)

    print("=" * 100)
    print("VALIDATION")
    print("=" * 100)
    print(f"input:  {args.input}")
    print(f"output: {out_path}")
    print(f"items:  {len(items)}")

    if all_errors:
        print()
        print("ERRORS:")
        for err in all_errors:
            print("-", err)
    else:
        print()
        print("OK: no validation errors")

    print_counter("QUESTION TYPES", Counter(item.get("question_type", "") for item in items))
    print_counter("DIFFICULTY", Counter(item.get("difficulty", "") for item in items))
    print_counter("EXPECTED BEHAVIOR", Counter(item.get("expected_behavior", "") for item in items))
    print_counter("CATEGORY", Counter(item.get("category", "") for item in items))
    print_counter("SOURCE DOCS", Counter(item.get("source_doc_id", "") for item in items))

    by_doc = defaultdict(int)
    by_chunk = defaultdict(int)
    for item in items:
        by_doc[item.get("source_doc_id", "")] += 1
        by_chunk[item.get("source_chunk_id", "")] += 1

    print()
    print("=" * 100)
    print("DOC/CHUNK COVERAGE")
    print("=" * 100)
    print(f"unique source_doc_id:   {len(by_doc)}")
    print(f"unique source_chunk_id: {len(by_chunk)}")

    print()
    print("=" * 100)
    print("LENGTH STATS")
    print("=" * 100)
    q_lens = [len(str(item.get("question", ""))) for item in items]
    a_lens = [len(str(item.get("expected_answer", ""))) for item in items]
    gp_lens = [len(item.get("gold_points", [])) for item in items if isinstance(item.get("gold_points"), list)]

    if items:
        print(f"avg question chars:        {sum(q_lens) / len(q_lens):.1f}")
        print(f"min/max question chars:    {min(q_lens)} / {max(q_lens)}")
        print(f"avg expected_answer chars: {sum(a_lens) / len(a_lens):.1f}")
        print(f"min/max answer chars:      {min(a_lens)} / {max(a_lens)}")
        print(f"avg gold_points count:     {sum(gp_lens) / len(gp_lens):.1f}")

    print()
    print("Saved fixed dataset:", out_path)


if __name__ == "__main__":
    main()
