from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.runtime import load_artifacts


def main():
    artifacts = load_artifacts()
    emb = np.asarray(artifacts.frida_embeddings)

    print("=" * 80)
    print("FRIDA EMBEDDINGS CHECK")
    print("=" * 80)

    print("shape:", emb.shape)
    print("dtype:", emb.dtype)

    nan_count = np.isnan(emb).sum()
    posinf_count = np.isposinf(emb).sum()
    neginf_count = np.isneginf(emb).sum()

    print("nan count:", int(nan_count))
    print("posinf count:", int(posinf_count))
    print("neginf count:", int(neginf_count))

    clean = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    print("min value:", float(clean.min()))
    print("max value:", float(clean.max()))
    print("max abs value:", float(np.abs(clean).max()))

    row_norms = np.linalg.norm(clean, axis=1)
    zero_norms = (row_norms == 0).sum()

    print("zero row norms:", int(zero_norms))
    print("row norm min:", float(row_norms.min()))
    print("row norm max:", float(row_norms.max()))
    print("row norm mean:", float(row_norms.mean()))

    bad_rows_mask = (
        np.isnan(emb).any(axis=1)
        | np.isinf(emb).any(axis=1)
    )
    bad_row_ids = np.where(bad_rows_mask)[0]

    print("bad rows count:", int(len(bad_row_ids)))
    if len(bad_row_ids) > 0:
        print("first bad row ids:", bad_row_ids[:20].tolist())

    huge_rows_mask = np.abs(clean).max(axis=1) > 1e6
    huge_row_ids = np.where(huge_rows_mask)[0]

    print("huge rows count (>1e6 abs):", int(len(huge_row_ids)))
    if len(huge_row_ids) > 0:
        print("first huge row ids:", huge_row_ids[:20].tolist())

    print("=" * 80)


if __name__ == "__main__":
    main()