from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


SRC = Path("dataset_gold.json")
BACKUP = Path("dataset_gold_before_auto_quote_fix.json")


def normalize_typographic_quotes(text: str) -> str:
    return (
        text.replace("\ufeff", "")
            .replace("“", '"')
            .replace("”", '"')
            .replace("„", '"')
            .replace("’", "'")
            .replace("‘", "'")
            .replace("\xa0", " ")
    )


def escape_inner_quotes(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '"':
            # если кавычка уже экранирована — не трогаем
            backslashes = 0
            j = i - 1
            while j >= 0 and s[j] == "\\":
                backslashes += 1
                j -= 1

            if backslashes % 2 == 1:
                out.append(ch)
            else:
                out.append('\\"')
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def fix_key_value_line(line: str) -> str:
    # Чинит строки вида:
    #   "source_title": "Приказ ... № 1161"Об утверждении..."",
    m = re.match(r'^(\s*"[^"]+"\s*:\s*")(.*)("\s*,?\s*)$', line)
    if not m:
        return line

    prefix, value, suffix = m.groups()
    return prefix + escape_inner_quotes(value) + suffix


def fix_array_string_line(line: str) -> str:
    # Чинит строки массива вида:
    #   "Приказ ... № 1161"Об утверждении..."",
    m = re.match(r'^(\s*")(.*)("\s*,?\s*)$', line)
    if not m:
        return line

    # Не трогаем key-value строки, их обрабатывает fix_key_value_line
    if re.match(r'^\s*"[^"]+"\s*:', line):
        return line

    prefix, value, suffix = m.groups()
    return prefix + escape_inner_quotes(value) + suffix


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(SRC)

    if not BACKUP.exists():
        shutil.copy2(SRC, BACKUP)

    text = SRC.read_text(encoding="utf-8-sig")
    text = normalize_typographic_quotes(text)

    fixed_lines = []
    for line in text.splitlines():
        line = fix_key_value_line(line)
        line = fix_array_string_line(line)
        fixed_lines.append(line)

    fixed = "\n".join(fixed_lines).strip() + "\n"

    # Проверяем, что теперь это настоящий JSON
    data = json.loads(fixed)

    if not isinstance(data, list):
        raise RuntimeError(f"Ожидался JSON-массив, получен: {type(data).__name__}")

    SRC.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("OK: dataset_gold.json fixed")
    print(f"items: {len(data)}")
    print(f"backup: {BACKUP}")


if __name__ == "__main__":
    main()
