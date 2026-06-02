from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence
import math

import pandas as pd

from app.retrieval.search import (
    search_frida,
    search_bm25_frida_weighted,
    search_bm25_frida_rrf,
    search_doc_first_top_chunks,
)

DEFAULT_KS = (1, 3, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on eval dataset")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/dataset_fixed.json"),
        help="Path to eval dataset json",
    )
    parser.add_argument(
    "--method",
    choices=["frida", "weighted", "rrf", "doc_first"],
    default="frida",
    help="Retrieval method",
    )

    parser.add_argument("--top-k", type=int, default=10, help="Max retrieval depth")
    parser.add_argument(
    "--doc-top-k",
    type=int,
    default=30,
    help="Number of top unique documents for doc_first search",
    )
    parser.add_argument(
    "--chunks-per-doc",
    type=int,
    default=3,
    help="Number of top cosine chunks selected from each document in doc_first search",
    )
    parser.add_argument(
    "--retrieval-top-k",
    type=int,
    default=300,
    help="First-stage retrieval depth for doc_first search",
    )
    parser.add_argument(
    "--base-search-mode",
    choices=["frida", "weighted", "rrf"],
    default="weighted",
    help="Base search mode used to select documents in doc_first search",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="*",
        default=list(DEFAULT_KS),
        help="Cutoffs to report, e.g. --ks 1 3 5 10",
    )
    parser.add_argument("--bm25-weight", type=float, default=0.3)
    parser.add_argument("--frida-weight", type=float, default=0.7)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--k-rrf", type=int, default=60)
    parser.add_argument(
        "--details-out",
        type=Path,
        default=None,
        help="Optional CSV path for per-query details",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional CSV path for overall summary",
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

def dcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(ranked[:k], start=1):
        rel_i = 1.0 if item in relevant else 0.0
        if rel_i > 0.0:
            dcg += rel_i / math.log2(i + 1)
    return dcg


def idcg_at_k(num_relevant: int, k: int) -> float:
    ideal_hits = min(num_relevant, k)
    idcg = 0.0
    for i in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(i + 1)
    return idcg


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = dcg_at_k(ranked, relevant, k)
    idcg = idcg_at_k(len(relevant), k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def first_hit_rank(ranked: Sequence[str], relevant: set[str]) -> int | None:
    for i, x in enumerate(ranked):
        if x in relevant:
            return i + 1
    return None

def retrieve(
    query: str,
    method: str,
    top_k: int,
    bm25_weight: float,
    frida_weight: float,
    candidate_k: int,
    k_rrf: int,
    doc_top_k: int,
    chunks_per_doc: int,
    retrieval_top_k: int,
    base_search_mode: str,
) -> pd.DataFrame:
    
    if method == "frida":
        return search_frida(query, top_k=top_k)

    if method == "weighted":
        return search_bm25_frida_weighted(
            query,
            top_k=top_k,
            bm25_weight=bm25_weight,
            frida_weight=frida_weight,
            candidate_k=candidate_k,
        )

    if method == "rrf":
        return search_bm25_frida_rrf(
            query,
            top_k=top_k,
            bm25_weight=bm25_weight,
            frida_weight=frida_weight,
            candidate_k=candidate_k,
            k_rrf=k_rrf,
        )

    if method == "doc_first":
        return search_doc_first_top_chunks(
            query=query,
            doc_top_k=doc_top_k,
            chunks_per_doc=chunks_per_doc,
            retrieval_top_k=retrieval_top_k,
            base_search_mode=base_search_mode,
            bm25_weight=bm25_weight,
            frida_weight=frida_weight,
            candidate_k=candidate_k,
            k_rrf=k_rrf,
        )

    raise ValueError(f"Unknown method: {method}")

# def retrieve(
#     query: str,
#     method: str,
#     top_k: int,
#     bm25_weight: float,
#     frida_weight: float,
#     candidate_k: int,
#     k_rrf: int,
# ) -> pd.DataFrame:
#     if method == "frida":
#         return search_frida(query, top_k=top_k)

#     if method == "weighted":
#         return search_bm25_frida_weighted(
#             query,
#             top_k=top_k,
#             bm25_weight=bm25_weight,
#             frida_weight=frida_weight,
#             candidate_k=candidate_k,
#         )

#     if method == "rrf":
#         return search_bm25_frida_rrf(
#             query,
#             top_k=top_k,
#             bm25_weight=bm25_weight,
#             frida_weight=frida_weight,
#             candidate_k=candidate_k,
#             k_rrf=k_rrf,
#         )

#     raise ValueError(f"Unknown method: {method}")


def evaluate_dataset(
    df: pd.DataFrame,
    method: str,
    top_k: int,
    ks: Sequence[int],
    bm25_weight: float,
    frida_weight: float,
    candidate_k: int,
    k_rrf: int,
    doc_top_k: int,
    chunks_per_doc: int,
    retrieval_top_k: int,
    base_search_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ks = sorted(set(int(k) for k in ks if 0 < int(k) <= top_k))
    rows = []

    for row in df.itertuples(index=False):
        results = retrieve(
            query=row.query,
            method=method,
            top_k=top_k,
            bm25_weight=bm25_weight,
            frida_weight=frida_weight,
            candidate_k=candidate_k,
            k_rrf=k_rrf,
            doc_top_k=doc_top_k,
            chunks_per_doc=chunks_per_doc,
            retrieval_top_k=retrieval_top_k,
            base_search_mode=base_search_mode,
        )

        retrieved_chunk_ids = results["chunk_id"].astype(str).tolist()
        retrieved_doc_ids = unique_preserve_order(results["doc_id"].astype(str).tolist())

        relevant_chunk_ids = set(map(str, row.relevant_chunk_ids))
        relevant_doc_ids = set(map(str, row.relevant_doc_ids))

        item = {
            "query": row.query,
            "query_type": row.query_type,
            "difficulty": row.difficulty,
            "category": row.category,
            "source_doc_id": row.source_doc_id,
            "source_chunk_id": row.source_chunk_id,
            "chunk_mrr": reciprocal_rank(retrieved_chunk_ids, relevant_chunk_ids),
            "doc_mrr": reciprocal_rank(retrieved_doc_ids, relevant_doc_ids),
            "first_chunk_hit": first_hit_rank(retrieved_chunk_ids, relevant_chunk_ids),
            "first_doc_hit": first_hit_rank(retrieved_doc_ids, relevant_doc_ids),
        }

        for k in ks:
            item[f"chunk_hit@{k}"] = hit_at_k(retrieved_chunk_ids, relevant_chunk_ids, k)
            item[f"doc_hit@{k}"] = hit_at_k(retrieved_doc_ids, relevant_doc_ids, k)
            item[f"chunk_ndcg@{k}"] = ndcg_at_k(retrieved_chunk_ids, relevant_chunk_ids, k)
            item[f"doc_ndcg@{k}"] = ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, k)

        rows.append(item)

    details = pd.DataFrame(rows)

    metric_cols = [c for c in details.columns if "@" in c or c.endswith("_mrr")]

    overall = pd.DataFrame([details[metric_cols].mean(numeric_only=True)]).assign(
    method=method,
    top_k=top_k,
    doc_top_k=doc_top_k,
    chunks_per_doc=chunks_per_doc,
    retrieval_top_k=retrieval_top_k,
    base_search_mode=base_search_mode,
    bm25_weight=bm25_weight,
    frida_weight=frida_weight,
    candidate_k=candidate_k,
    k_rrf=k_rrf,
    n_queries=len(details),
    )

    by_type = (
        details.groupby("query_type")[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("query_type")
        .reset_index(drop=True)
    )

    by_difficulty = (
        details.groupby("difficulty")[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("difficulty")
        .reset_index(drop=True)
    )

    by_category = (
        details.groupby("category")[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("category")
        .reset_index(drop=True)
    )

    return details, overall, by_type, by_difficulty, by_category


def main() -> None:
    args = parse_args()
    df = load_dataset(args.dataset)

    details, overall, by_type, by_difficulty, by_category = evaluate_dataset(
        df=df,
        method=args.method,
        top_k=args.top_k,
        ks=args.ks,
        bm25_weight=args.bm25_weight,
        frida_weight=args.frida_weight,
        candidate_k=args.candidate_k,
        k_rrf=args.k_rrf,
        doc_top_k=args.doc_top_k,
        chunks_per_doc=args.chunks_per_doc,
        retrieval_top_k=args.retrieval_top_k,
        base_search_mode=args.base_search_mode,
    )

    print("=" * 100)
    print("OVERALL")
    print("=" * 100)
    print(overall.to_string(index=False))

    print("\n" + "=" * 100)
    print("BY QUERY TYPE")
    print("=" * 100)
    print(by_type.to_string(index=False))

    print("\n" + "=" * 100)
    print("BY DIFFICULTY")
    print("=" * 100)
    print(by_difficulty.to_string(index=False))

    print("\n" + "=" * 100)
    print("BY CATEGORY")
    print("=" * 100)
    print(by_category.to_string(index=False))

    print("\n" + "=" * 100)
    print("SAMPLE DETAILS")
    print("=" * 100)
    
    sample_cols = [
        "query_type",
        "difficulty",
        "category",
        "source_doc_id",
        "source_chunk_id",
        "chunk_mrr",
        "doc_mrr",
    ] + [
        c for c in details.columns
        if c.startswith("chunk_hit@")
        or c.startswith("doc_hit@")
        or c.startswith("chunk_ndcg@")
        or c.startswith("doc_ndcg@")
    ]
    
    print(details[sample_cols].head(10).to_string(index=False))

    if args.details_out:
        args.details_out.parent.mkdir(parents=True, exist_ok=True)
        details.to_csv(args.details_out, index=False)
        print(f"\nSaved details to: {args.details_out}")

    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        overall.to_csv(args.summary_out, index=False)
        print(f"Saved summary to: {args.summary_out}")


if __name__ == "__main__":
    main()