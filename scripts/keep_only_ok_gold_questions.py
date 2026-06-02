from __future__ import annotations

import json
from pathlib import Path


SRC = Path("dataset_gold.json")
DST = Path("dataset_gold_ok_only.json")


BAD_OR_REVISE_IDS = {
    "qa007",
    "qa012",
    "qa013",
    "qa014",
    "qa016",
    "qa017",
    "qa018",
    "qa024",
    "qa025",
    "qa026",
    "qa029",
    "qa034",
    "qa035",
    "qa036",
    "qa046",
    "qa047",
    "qa048",
    "qa049",
    "qa050",
    "qa057",
    "qa058",
    "qa065",
}


def find_matching_brace(text: str, start: int) -> int | None:
    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i

    return None


def extract_objects_from_possibly_broken_json(text: str) -> list[dict]:
    items: list[dict] = []
    pos = 0

    while True:
        qpos = text.find('"id"', pos)
        if qpos == -1:
            break

        start = text.rfind("{", 0, qpos)
        if start == -1:
            pos = qpos + 4
            continue

        end = find_matching_brace(text, start)
        if end is None:
            pos = qpos + 4
            continue

        raw = text[start:end + 1]

        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"SKIP broken object near char {start}: {e}")
            pos = end + 1
            continue

        if isinstance(obj, dict) and obj.get("id"):
            items.append(obj)

        pos = end + 1

    return items


def main() -> None:
    text = SRC.read_text(encoding="utf-8-sig")
    items = extract_objects_from_possibly_broken_json(text)

    if not items:
        raise RuntimeError("Не удалось извлечь вопросы из dataset_gold.json")

    before = len(items)

    kept = []
    removed = []

    for item in items:
        item_id = item.get("id", "")
        if item_id in BAD_OR_REVISE_IDS:
            removed.append(item)
        else:
            kept.append(item)

    # Перенумеруем оставшиеся id подряд, чтобы не было дыр.
    for i, item in enumerate(kept, start=1):
        item["id"] = f"qa{i:03d}"

    DST.write_text(
        json.dumps(kept, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 100)
    print("FILTER OK ONLY")
    print("=" * 100)
    print(f"source:  {SRC}")
    print(f"output:  {DST}")
    print(f"before:  {before}")
    print(f"kept:    {len(kept)}")
    print(f"removed: {len(removed)}")

    print()
    print("REMOVED IDS:")
    for x in removed:
        print(x.get("id"), "|", x.get("question"))

    print()
    print("KEPT IDS AFTER RENUMBER:")
    for x in kept:
        print(x.get("id"), "|", x.get("source_doc_id"), "|", x.get("question")[:120])


if __name__ == "__main__":
    main()
