from pathlib import Path
import sys
import json
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.runtime import load_artifacts


def main():
    dataset_path = PROJECT_ROOT / "data" / "processed" / "dataset_fixed.json"

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    artifacts = load_artifacts()
    encoder = artifacts.frida_encoder

    print("=" * 80)
    print("QUERY EMBEDDINGS CHECK")
    print("=" * 80)
    print("dataset:", dataset_path)
    print("n queries:", len(data))

    bad_queries = []
    huge_queries = []
    zero_norm_queries = []

    for i, row in enumerate(data):
        query = row["query"]

        try:
            q_emb = encoder.encode(
                query,
                prompt_name="search_query",
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        except Exception as e:
            bad_queries.append({
                "ix": i,
                "query": query,
                "error": repr(e),
            })
            continue

        q_emb = np.asarray(q_emb)

        has_nan = np.isnan(q_emb).any()
        has_posinf = np.isposinf(q_emb).any()
        has_neginf = np.isneginf(q_emb).any()

        clean = np.nan_to_num(q_emb, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        norm = float(np.linalg.norm(clean))
        max_abs = float(np.abs(clean).max())

        if has_nan or has_posinf or has_neginf:
            bad_queries.append({
                "ix": i,
                "query": query,
                "has_nan": bool(has_nan),
                "has_posinf": bool(has_posinf),
                "has_neginf": bool(has_neginf),
                "norm": norm,
                "max_abs": max_abs,
            })

        if norm == 0.0:
            zero_norm_queries.append({
                "ix": i,
                "query": query,
            })

        if max_abs > 1e6:
            huge_queries.append({
                "ix": i,
                "query": query,
                "norm": norm,
                "max_abs": max_abs,
            })

    print("bad queries:", len(bad_queries))
    if bad_queries:
        print("\nFIRST BAD QUERIES:")
        for x in bad_queries[:10]:
            print(x)

    print("\nzero norm queries:", len(zero_norm_queries))
    if zero_norm_queries:
        print("\nFIRST ZERO NORM QUERIES:")
        for x in zero_norm_queries[:10]:
            print(x)

    print("\nhuge queries (>1e6 abs):", len(huge_queries))
    if huge_queries:
        print("\nFIRST HUGE QUERIES:")
        for x in huge_queries[:10]:
            print(x)

    print("=" * 80)


if __name__ == "__main__":
    main()