from __future__ import annotations

import os
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def fail(msg: str) -> None:
    print(f"[ERROR] {msg}")
    sys.exit(1)


def has_model_files(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False

    markers = [
        "modules.json",
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
    ]

    return any((path / x).exists() for x in markers)


def main() -> None:
    model_id = os.getenv("FRIDA_MODEL_ID", "ai-forever/FRIDA").strip()
    model_path = Path(os.getenv("FRIDA_MODEL_PATH", "models/FRIDA")).resolve()

    print("=" * 100)
    print("CHECK / DOWNLOAD FRIDA MODEL")
    print("=" * 100)
    print(f"FRIDA_MODEL_ID:   {model_id}")
    print(f"FRIDA_MODEL_PATH: {model_path}")

    if has_model_files(model_path):
        ok(f"FRIDA model already exists: {model_path}")
        return

    if not model_id:
        fail("FRIDA_MODEL_ID is empty and local model is missing")

    model_path.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] downloading/loading model via sentence-transformers")
    print(f"[INFO] this may take time on first run: {model_id}")

    try:
        model = SentenceTransformer(model_id)
    except Exception as e:
        fail(f"cannot load model '{model_id}' via SentenceTransformer: {e}")

    print(f"[INFO] saving model to: {model_path}")
    try:
        model.save(str(model_path))
    except Exception as e:
        fail(f"cannot save model to '{model_path}': {e}")

    if not has_model_files(model_path):
        fail(f"model was saved, but expected model files were not found in: {model_path}")

    ok(f"FRIDA model installed into: {model_path}")


if __name__ == "__main__":
    main()
