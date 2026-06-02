from __future__ import annotations

import json
import re
from pathlib import Path


SRC = Path("dataset_gold.json")
DST = Path("dataset_gold_repaired.json")


def extract_json_values(text: str):
    decoder = json.JSONDecoder()
    values = []
    i = 0

    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1

        if i >= len(text):
            break

        # пропускаем мусорные разделители между вставками
        if text[i] in ",;":
            i += 1
            continue

        # пробуем распарсить JSON value с текущей позиции
        try:
            value, end = decoder.raw_decode(text, i)
            values.append(value)
            i = end
            continue
        except json.JSONDecodeError:
            # если встретили лишнюю внешнюю скобку/мусор — двигаемся дальше
            i += 1

    return values


def flatten(values):
    items = []

    for value in values:
        if isinstance(value, list):
            for x in value:
                if isinstance(x, dict):
                    if x.get("resample") is True:
                        continue
                    items.append(x)
        elif isinstance(value, dict):
            if value.get("resample") is True:
                continue
            items.append(value)

    return items


def main():
    text = SRC.read_text(encoding="utf-8-sig")

    # убрать markdown fences, если случайно попали
    text = re.sub(r"```(?:json)?", "", text)
    text = text.replace("```", "")

    values = extract_json_values(text)
    items = flatten(values)

    if not items:
        raise RuntimeError("Не удалось извлечь ни одного JSON-объекта. Покажи первые 40 строк файла.")

    DST.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"extracted json values: {len(values)}")
    print(f"dataset items: {len(items)}")
    print(f"saved: {DST}")


if __name__ == "__main__":
    main()
