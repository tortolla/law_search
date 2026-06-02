import requests
from pathlib import Path
from tqdm import tqdm

PUBLIC_URL = "https://disk.yandex.ru/d/IcLlGxelh0A8GQ"
OUT_DIR = Path("data_big")
API_URL = "https://cloud-api.yandex.net/v1/disk/public/resources"

ROOT_DIR_SUFFIX = "_laws"


def get_resource(path="", offset=0, limit=1000):
    params = {
        "public_key": PUBLIC_URL,
        "path": path,
        "limit": limit,
        "offset": offset,
    }
    r = requests.get(API_URL, params=params)
    r.raise_for_status()
    return r.json()


def get_all_items(path=""):
    all_items = []
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


def rel_path(yandex_path):
    return yandex_path.replace("disk:", "").lstrip("/")


def download_file(file_path):
    out_path = OUT_DIR / rel_path(file_path)

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"Уже есть, пропускаю: {out_path}")
        return

    params = {
        "public_key": PUBLIC_URL,
        "path": file_path,
    }

    meta = requests.get(API_URL, params=params)
    meta.raise_for_status()
    download_url = meta.json().get("file")

    if not download_url:
        print(f"Нет ссылки на скачивание: {file_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))

        with open(out_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=str(out_path),
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def download_dir_recursive(dir_path):
    """
    Полностью скачивает папку.
    Используется только для *_md папок.
    """
    for item in get_all_items(dir_path):
        item_path = item["path"]

        if item["type"] == "file":
            download_file(item_path)

        elif item["type"] == "dir":
            download_dir_recursive(item_path)


def scan_laws_dir(path):
    """
    Сканирует только одну верхнеуровневую папку *_laws.

    Внутри скачивает:
    - metadata.json
    - hierarchy.json
    - папки *_md полностью

    Папки без _md внутри *_laws пропускаются.
    """
    for item in get_all_items(path):
        item_type = item["type"]
        item_name = item["name"]
        item_path = item["path"]

        if item_type == "file":
            if item_name in {"metadata.json", "hierarchy.json"}:
                download_file(item_path)

        elif item_type == "dir":
            if item_name.endswith("_md"):
                print(f"Скачиваю md-папку полностью: {item_path}")
                download_dir_recursive(item_path)
            else:
                print(f"Пропускаю папку без _md: {item_path}")


def scan_root():
    """
    Сканирует верхний уровень Яндекс.Диска.

    Берёт только папки, которые заканчиваются на *_laws.
    Например:
    - energy_laws скачиваем
    - mining_laws скачиваем
    - energy НЕ скачиваем
    """
    for item in get_all_items(""):
        item_type = item["type"]
        item_name = item["name"]
        item_path = item["path"]

        if item_type != "dir":
            continue

        if not item_name.endswith(ROOT_DIR_SUFFIX):
            print(f"Пропускаю верхнеуровневую папку не *_laws: {item_path}")
            continue

        print(f"Обрабатываю папку законов: {item_path}")
        scan_laws_dir(item_path)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scan_root()
    print("Готово: скачаны только *_laws папки, внутри них только *_md и нужные json.")