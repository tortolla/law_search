import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from app.retrieval.config import (
    CHUNKS_PATH,
    FRIDA_DIR,
    FRIDA_EMBEDDINGS_PATH,
    FRIDA_INFO_PATH,
    FRIDA_MODEL_PATH,
)


def main():
    FRIDA_DIR.mkdir(parents=True, exist_ok=True)

    chunks_df = pd.read_parquet(CHUNKS_PATH)
    texts = chunks_df["chunk_text"].fillna("").tolist()

    model = SentenceTransformer(str(FRIDA_MODEL_PATH))
    print(f"Loaded FRIDA model from: {FRIDA_MODEL_PATH}")

    embeddings = model.encode(
        texts,
        prompt_name="search_document",
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.save(FRIDA_EMBEDDINGS_PATH, embeddings)

    info = {
        "model_path": str(FRIDA_MODEL_PATH),
        "num_chunks": int(len(texts)),
        "embedding_dim": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
    }

    with open(FRIDA_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"[OK] embeddings -> {FRIDA_EMBEDDINGS_PATH}")
    print(f"[OK] info -> {FRIDA_INFO_PATH}")
    print(info)


if __name__ == "__main__":
    main()