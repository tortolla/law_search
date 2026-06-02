from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import matplotlib.pyplot as plt

from app.retrieval.search import (
    search_frida,
    search_bm25_frida_weighted,
    search_bm25_frida_rrf,
)
from app.retrieval.runtime import load_artifacts


DEFAULT_KS = (1, 3, 5, 10, 20, 30, 40, 50, 60, 100, 200, 300)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate document-level retrieval on ground truth dataset"
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("ground_truth_v4.json"),
        help="Path to ground truth json",
    )

    parser.add_argument(
        "--method",
        choices=["frida", "weighted", "rrf"],
        default="weighted",
        help="Retrieval method",
    )

    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--ks", type=int, nargs="*", default=list(DEFAULT_KS))

    parser.add_argument("--bm25-weight", type=float, default=0.3)
    parser.add_argument("--frida-weight", type=float, default=0.7)
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--k-rrf", type=int, default=60)

    parser.add_argument(
        "--details-out",
        type=Path,
        default=Path("results/ground_truth_doc_details.csv"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("results/ground_truth_doc_summary.csv"),
    )
    parser.add_argument(
        "--missing-docs-out",
        type=Path,
        default=Path("results/ground_truth_missing_docs.csv"),
    )
    parser.add_argument(
        "--failed-out",
        type=Path,
        default=Path("results/ground_truth_failed_queries.csv"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("results/figures"),
    )

    return parser.parse_args()


def parse_doc_ids(value) -> list[str]:
    """
    source_doc_id can be:
    - "0001201807250005"
    - "3801202001160005, 3800202204290009"
    - list[str]
    """
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    text = str(value).strip()
    if not text:
        return []

    return [x.strip() for x in text.split(",") if x.strip()]


def normalize_list_value(value) -> str:
    if isinstance(value, list):
        return "; ".join(map(str, value))
    return str(value)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(data)

    required_cols = ["id", "question", "source_doc_id"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    optional_cols = [
        "source_title",
        "query_type",
        "difficulty",
        "category",
        "labeling",
    ]

    for col in optional_cols:
        if col not in df.columns:
            df[col] = ""

    keep_cols = required_cols + optional_cols
    df = df[keep_cols].copy()

    df["question"] = df["question"].astype(str)
    df["source_doc_ids_parsed"] = df["source_doc_id"].apply(parse_doc_ids)
    df["source_doc_ids_norm"] = df["source_doc_ids_parsed"].apply(lambda xs: "; ".join(xs))

    for col in ["query_type", "category"]:
        df[col] = df[col].apply(normalize_list_value)

    return df


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    out = []

    for x in items:
        x = str(x)
        if x not in seen:
            seen.add(x)
            out.append(x)

    return out


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return float(any(x in relevant for x in ranked[:k]))


def first_hit_rank(ranked: Sequence[str], relevant: set[str]) -> int | None:
    for i, x in enumerate(ranked, start=1):
        if x in relevant:
            return i
    return None


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    rank = first_hit_rank(ranked, relevant)
    if rank is None:
        return 0.0
    return 1.0 / rank


def dcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = 0.0

    for i, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 1)

    return dcg


def idcg_at_k(num_relevant: int, k: int) -> float:
    ideal_hits = min(num_relevant, k)
    return sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0

    idcg = idcg_at_k(len(relevant), k)
    if idcg == 0.0:
        return 0.0

    return dcg_at_k(ranked, relevant, k) / idcg


def retrieve(
    query: str,
    method: str,
    top_k: int,
    bm25_weight: float,
    frida_weight: float,
    candidate_k: int,
    k_rrf: int,
) -> pd.DataFrame:
    if method == "frida":
        return search_frida(
            query=query,
            top_k=top_k,
        )

    if method == "weighted":
        return search_bm25_frida_weighted(
            query=query,
            top_k=top_k,
            bm25_weight=bm25_weight,
            frida_weight=frida_weight,
            candidate_k=candidate_k,
        )

    if method == "rrf":
        return search_bm25_frida_rrf(
            query=query,
            top_k=top_k,
            bm25_weight=bm25_weight,
            frida_weight=frida_weight,
            candidate_k=candidate_k,
            k_rrf=k_rrf,
        )

    raise ValueError(f"Unknown method: {method}")


def get_index_doc_ids() -> set[str]:
    artifacts = load_artifacts()
    chunks_df = artifacts.chunks_df
    return set(chunks_df["doc_id"].astype(str).unique())


def build_missing_docs_report(df: pd.DataFrame, index_doc_ids: set[str]) -> pd.DataFrame:
    rows = []

    for row in df.itertuples(index=False):
        for doc_id in row.source_doc_ids_parsed:
            rows.append(
                {
                    "id": row.id,
                    "question": row.question,
                    "source_doc_id": doc_id,
                    "source_title": row.source_title,
                    "is_present_in_index": doc_id in index_doc_ids,
                    "query_type": row.query_type,
                    "difficulty": row.difficulty,
                    "category": row.category,
                    "labeling": row.labeling,
                }
            )

    return pd.DataFrame(rows)


def evaluate(
    df: pd.DataFrame,
    method: str,
    top_k: int,
    ks: Sequence[int],
    bm25_weight: float,
    frida_weight: float,
    candidate_k: int,
    k_rrf: int,
    index_doc_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ks = sorted(set(int(k) for k in ks if 0 < int(k) <= top_k))

    rows = []

    for row in df.itertuples(index=False):
        query = str(row.question)
        relevant_doc_ids = set(row.source_doc_ids_parsed)

        present_relevant_doc_ids = relevant_doc_ids & index_doc_ids
        missing_relevant_doc_ids = relevant_doc_ids - index_doc_ids

        results = retrieve(
            query=query,
            method=method,
            top_k=top_k,
            bm25_weight=bm25_weight,
            frida_weight=frida_weight,
            candidate_k=candidate_k,
            k_rrf=k_rrf,
        )

        retrieved_chunk_ids = results["chunk_id"].astype(str).tolist() if "chunk_id" in results.columns else []
        retrieved_doc_ids_raw = results["doc_id"].astype(str).tolist()
        retrieved_doc_ids = unique_preserve_order(retrieved_doc_ids_raw)

        first_rank = first_hit_rank(retrieved_doc_ids, relevant_doc_ids)
        first_rank_present_only = first_hit_rank(retrieved_doc_ids, present_relevant_doc_ids)

        item = {
            "id": row.id,
            "question": query,
            "source_doc_id": row.source_doc_id,
            "source_doc_ids_norm": row.source_doc_ids_norm,
            "source_title": row.source_title,
            "query_type": row.query_type,
            "difficulty": row.difficulty,
            "category": row.category,
            "labeling": row.labeling,

            "n_relevant_docs": len(relevant_doc_ids),
            "n_present_relevant_docs": len(present_relevant_doc_ids),
            "n_missing_relevant_docs": len(missing_relevant_doc_ids),
            "missing_relevant_doc_ids": "; ".join(sorted(missing_relevant_doc_ids)),

            "n_retrieved_chunks": len(retrieved_chunk_ids),
            "n_retrieved_unique_docs": len(retrieved_doc_ids),

            "first_doc_hit": first_rank,
            "first_doc_hit_present_only": first_rank_present_only,

            "doc_mrr": reciprocal_rank(retrieved_doc_ids, relevant_doc_ids),
            "doc_mrr_present_only": reciprocal_rank(retrieved_doc_ids, present_relevant_doc_ids),

            "retrieved_doc_ids_top": "; ".join(retrieved_doc_ids[:top_k]),
        }

        for k in ks:
            item[f"doc_hit@{k}"] = hit_at_k(retrieved_doc_ids, relevant_doc_ids, k)
            item[f"doc_ndcg@{k}"] = ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, k)

            item[f"doc_hit_present_only@{k}"] = hit_at_k(
                retrieved_doc_ids,
                present_relevant_doc_ids,
                k,
            )
            item[f"doc_ndcg_present_only@{k}"] = ndcg_at_k(
                retrieved_doc_ids,
                present_relevant_doc_ids,
                k,
            )

        rows.append(item)

    details = pd.DataFrame(rows)

    metric_cols = [
        c for c in details.columns
        if c.startswith("doc_hit@")
        or c.startswith("doc_ndcg@")
        or c.startswith("doc_hit_present_only@")
        or c.startswith("doc_ndcg_present_only@")
        or c in ["doc_mrr", "doc_mrr_present_only"]
    ]

    summary = pd.DataFrame([details[metric_cols].mean(numeric_only=True)]).assign(
        method=method,
        top_k=top_k,
        bm25_weight=bm25_weight,
        frida_weight=frida_weight,
        candidate_k=candidate_k,
        k_rrf=k_rrf,
        n_queries=len(details),
        n_queries_with_all_gold_docs_present=int((details["n_missing_relevant_docs"] == 0).sum()),
        n_queries_with_missing_gold_docs=int((details["n_missing_relevant_docs"] > 0).sum()),
        avg_retrieved_unique_docs=details["n_retrieved_unique_docs"].mean(),
        min_retrieved_unique_docs=details["n_retrieved_unique_docs"].min(),
        max_retrieved_unique_docs=details["n_retrieved_unique_docs"].max(),
    )

    return details, summary


def plot_metric(
    summary: pd.DataFrame,
    prefix: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    cols = []

    for col in summary.columns:
        if col.startswith(prefix + "@"):
            k = int(col.split("@")[1])
            cols.append((k, col))

    cols.sort(key=lambda x: x[0])

    if not cols:
        print(f"No columns for prefix: {prefix}")
        return

    ks = [k for k, _ in cols]
    values = [float(summary.iloc[0][col]) for _, col in cols]

    plt.figure(figsize=(10, 5.5))
    plt.plot(ks, values, marker="o", linewidth=2)
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel(ylabel)
    plt.xticks(ks)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved figure: {out_path}")


def print_diagnostics(details: pd.DataFrame, summary: pd.DataFrame) -> None:
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(summary.to_string(index=False))

    print("\n" + "=" * 100)
    print("RETRIEVED UNIQUE DOCS DISTRIBUTION")
    print("=" * 100)
    print(details["n_retrieved_unique_docs"].describe().to_string())

    print("\n" + "=" * 100)
    print("GOLD DOCS COVERAGE")
    print("=" * 100)
    print("queries:", len(details))
    print("queries with missing gold docs:", int((details["n_missing_relevant_docs"] > 0).sum()))
    print("queries with all gold docs present:", int((details["n_missing_relevant_docs"] == 0).sum()))

    hit_cols = [c for c in summary.columns if c.startswith("doc_hit@")]
    ndcg_cols = [c for c in summary.columns if c.startswith("doc_ndcg@")]

    print("\n" + "=" * 100)
    print("DOC HIT")
    print("=" * 100)
    print(summary[hit_cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("DOC NDCG")
    print("=" * 100)
    print(summary[ndcg_cols].to_string(index=False))


def main() -> None:
    args = parse_args()

    df = load_dataset(args.dataset)
    index_doc_ids = get_index_doc_ids()

    details, summary = evaluate(
        df=df,
        method=args.method,
        top_k=args.top_k,
        ks=args.ks,
        bm25_weight=args.bm25_weight,
        frida_weight=args.frida_weight,
        candidate_k=args.candidate_k,
        k_rrf=args.k_rrf,
        index_doc_ids=index_doc_ids,
    )

    missing_docs = build_missing_docs_report(df, index_doc_ids)

    failed = details[details[f"doc_hit@{max([k for k in args.ks if k <= args.top_k])}"] == 0].copy()

    args.details_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.missing_docs_out.parent.mkdir(parents=True, exist_ok=True)
    args.failed_out.parent.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    details.to_csv(args.details_out, index=False)
    summary.to_csv(args.summary_out, index=False)
    missing_docs.to_csv(args.missing_docs_out, index=False)
    failed.to_csv(args.failed_out, index=False)

    print_diagnostics(details, summary)

    print("\nSaved details:", args.details_out)
    print("Saved summary:", args.summary_out)
    print("Saved missing docs report:", args.missing_docs_out)
    print("Saved failed queries:", args.failed_out)

    plot_metric(
        summary=summary,
        prefix="doc_hit",
        title="Document Hit@k / Accuracy@k",
        ylabel="DocHit",
        out_path=args.figures_dir / "ground_truth_doc_hit_at_k.png",
    )

    plot_metric(
        summary=summary,
        prefix="doc_ndcg",
        title="Document NDCG@k",
        ylabel="DocNDCG",
        out_path=args.figures_dir / "ground_truth_doc_ndcg_at_k.png",
    )

    plot_metric(
        summary=summary,
        prefix="doc_hit_present_only",
        title="Document Hit@k / Present Gold Docs Only",
        ylabel="DocHit",
        out_path=args.figures_dir / "ground_truth_doc_hit_present_only_at_k.png",
    )

    plot_metric(
        summary=summary,
        prefix="doc_ndcg_present_only",
        title="Document NDCG@k / Present Gold Docs Only",
        ylabel="DocNDCG",
        out_path=args.figures_dir / "ground_truth_doc_ndcg_present_only_at_k.png",
    )


if __name__ == "__main__":
    main()
