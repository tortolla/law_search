from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from app.retrieval.search import search_bm25_frida_weighted


DEFAULT_KS = (1, 3, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search for weighted BM25 + FRIDA retrieval")

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/dataset_fixed.json"),
        help="Path to eval dataset json",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Retrieval depth")
    parser.add_argument(
        "--ks",
        type=int,
        nargs="*",
        default=[1, 3],
        help="Cutoffs to report, e.g. --ks 1 3 5",
    )
    parser.add_argument(
        "--bm25-grid",
        type=float,
        nargs="*",
        default=[round(x * 0.05, 2) for x in range(21)],
        help="Grid of BM25 weights; FRIDA weight is 1 - BM25 weight",
    )
    parser.add_argument(
        "--candidate-ks",
        type=int,
        nargs="*",
        default=[100, 200, 400],
        help="Grid of candidate_k values",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="chunk_hit@3",
        help="Metric to sort final results by",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/grid_search_weighted.csv"),
        help="Where to save grid search results CSV",
    )

    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(data)

    required_cols = [
        "query",
        "query_type",
        "difficulty",
        "source_doc_id",
        "source_chunk_id",
        "category",
        "relevant_doc_ids",
        "relevant_chunk_ids",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    return df


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return float(any(x in relevant for x in ranked[:k]))


def evaluate_one_config(
    df: pd.DataFrame,
    top_k: int,
    ks: Sequence[int],
    bm25_weight: float,
    frida_weight: float,
    candidate_k: int,
) -> dict:
    ks = sorted(set(int(k) for k in ks if 0 < int(k) <= top_k))

    rows = []

    for row in df.itertuples(index=False):
        results = search_bm25_frida_weighted(
            row.query,
            top_k=top_k,
            bm25_weight=bm25_weight,
            frida_weight=frida_weight,
            candidate_k=candidate_k,
        )

        retrieved_chunk_ids = results["chunk_id"].astype(str).tolist()
        retrieved_doc_ids = unique_preserve_order(results["doc_id"].astype(str).tolist())

        relevant_chunk_ids = set(map(str, row.relevant_chunk_ids))
        relevant_doc_ids = set(map(str, row.relevant_doc_ids))

        item = {
            "chunk_mrr": reciprocal_rank(retrieved_chunk_ids, relevant_chunk_ids),
            "doc_mrr": reciprocal_rank(retrieved_doc_ids, relevant_doc_ids),
        }

        for k in ks:
            item[f"chunk_hit@{k}"] = hit_at_k(retrieved_chunk_ids, relevant_chunk_ids, k)
            item[f"doc_hit@{k}"] = hit_at_k(retrieved_doc_ids, relevant_doc_ids, k)

        rows.append(item)

    details = pd.DataFrame(rows)
    metrics = details.mean(numeric_only=True).to_dict()

    out = {
        "bm25_weight": bm25_weight,
        "frida_weight": frida_weight,
        "candidate_k": candidate_k,
        "top_k": top_k,
        "n_queries": len(df),
        **metrics,
    }
    return out


def main() -> None:
    args = parse_args()
    df = load_dataset(args.dataset)

    results = []

    print("=" * 100)
    print("GRID SEARCH WEIGHTED")
    print("=" * 100)
    print("dataset:", args.dataset)
    print("top_k:", args.top_k)
    print("ks:", args.ks)
    print("bm25_grid:", args.bm25_grid)
    print("candidate_ks:", args.candidate_ks)
    print()

    total = len(args.bm25_grid) * len(args.candidate_ks)
    done = 0

    for bm25_weight in args.bm25_grid:
        frida_weight = round(1.0 - bm25_weight, 10)

        for candidate_k in args.candidate_ks:
            done += 1
            print(
                f"[{done}/{total}] "
                f"bm25={bm25_weight:.2f} "
                f"frida={frida_weight:.2f} "
                f"candidate_k={candidate_k}"
            )

            row = evaluate_one_config(
                df=df,
                top_k=args.top_k,
                ks=args.ks,
                bm25_weight=bm25_weight,
                frida_weight=frida_weight,
                candidate_k=candidate_k,
            )
            results.append(row)

    result_df = pd.DataFrame(results)

    sort_by = args.sort_by
    if sort_by not in result_df.columns:
        raise ValueError(
            f"sort-by column '{sort_by}' not found. Available columns: {list(result_df.columns)}"
        )

    result_df = result_df.sort_values(
        by=[sort_by, "chunk_mrr", "doc_hit@3" if "doc_hit@3" in result_df.columns else "doc_mrr"],
        ascending=False,
    ).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.out, index=False)

    print("\n" + "=" * 100)
    print("TOP RESULTS")
    print("=" * 100)
    cols_to_show = [
        "bm25_weight",
        "frida_weight",
        "candidate_k",
        "chunk_mrr",
        "doc_mrr",
    ] + [c for c in result_df.columns if c.startswith("chunk_hit@") or c.startswith("doc_hit@")]
    cols_to_show = [c for c in cols_to_show if c in result_df.columns]
    print(result_df[cols_to_show].head(15).to_string(index=False))

    print(f"\nSaved grid search results to: {args.out}")


if __name__ == "__main__":
    main()