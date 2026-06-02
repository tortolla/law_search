import pickle
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from app.retrieval.config import (
    CHUNKS_PATH,
    BM25_PATH,
    FRIDA_EMBEDDINGS_PATH,
    FRIDA_MODEL_PATH,
)


@dataclass
class RetrievalArtifacts:
    chunks_df: pd.DataFrame
    bm25: object
    frida_embeddings: np.ndarray
    frida_encoder: SentenceTransformer


def _check_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Не найден артефакт: {path}")


@lru_cache(maxsize=1)
def load_artifacts() -> RetrievalArtifacts:
    _check_file(CHUNKS_PATH)
    _check_file(BM25_PATH)
    _check_file(FRIDA_EMBEDDINGS_PATH)
    _check_file(FRIDA_MODEL_PATH)

    chunks_df = pd.read_parquet(CHUNKS_PATH)

    with open(BM25_PATH, "rb") as f:
        bm25_payload = pickle.load(f)

    bm25 = bm25_payload["bm25"]
    frida_embeddings = np.load(FRIDA_EMBEDDINGS_PATH, mmap_mode="r")
    frida_encoder = SentenceTransformer(str(FRIDA_MODEL_PATH))

    if len(chunks_df) != len(frida_embeddings):
        raise ValueError(
            f"Несовпадение артефактов: chunks={len(chunks_df)} embeddings={len(frida_embeddings)}"
        )

    return RetrievalArtifacts(
        chunks_df=chunks_df,
        bm25=bm25,
        frida_embeddings=frida_embeddings,
        frida_encoder=frida_encoder,
    )