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


DEFAULT_KS = (1, 3, 5, 10, 20, 30, 40, 50, 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate doc retrieval on ground truth dataset")

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
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--ks", type=int, nargs="*", default=list(DEFAULT_KS))

    parser.add_argument("--bm25-weight", type=float, default=0.3)
    parser.add_argument("--frida-weight", type=float, default=0.7)
    parser.add_argument("--candidate-k", type=int, default=500)
    parser.add_argument("--k-rrf", type=int, default=60)

    parser.add_argument(
        "--details-out",
        type=Path,
        default=Path("results/ground_truth_weighted_doc_details.csv"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("results/ground_truth_weighted_doc_summary.csv"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("results/figures"),
    )

    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(data)

    required = ["id", "question", "source_doc_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[[
        "id",
        "question",
        "source_doc_id",
        "source_title",
        "query_type",
        "difficulty",
        "category",
        "labeling",
    ]].copy()

    return df


def parse_doc_ids(value) -> list[str]:
    """
    source_doc_id может быть:
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
        return search_frida(query, top_k=top_k)

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


def evaluate(
    df: pd.DataFrame,
    method: str,
    top_k: int,
    ks: Sequence[int],
    bm25_weight: float,
    frida_weight: float,
    candidate_k: int,
    k_rrf: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ks = sorted(set(int(k) for k in ks if 0 < int(k) <= top_k))

    rows = []

    for row in df.itertuples(index=False):
        query = str(row.question)
        relevant_doc_ids = set(parse_doc_ids(row.source_doc_id))

        results = retrieve(
            query=query,
            method=method,
            top_k=top_k,
            bm25_weight=bm25_weight,
            frida_weight=frida_weight,
            candidate_k=candidate_k,
            k_rrf=k_rrf,
        )

        retrieved_doc_ids = unique_preserve_order(
            results["doc_id"].astype(str).tolist()
        )

        first_rank = first_hit_rank(retrieved_doc_ids, relevant_doc_ids)

        item = {
            "id": row.id,
            "question": query,
            "source_doc_id": row.source_doc_id,
            "source_title": row.source_title,
            "query_type": row.query_type,
            "difficulty": row.difficulty,
            "category": row.category,
            "labeling": row.labeling,
            "n_relevant_docs": len(relevant_doc_ids),
            "first_doc_hit": first_rank,
            "doc_mrr": reciprocal_rank(retrieved_doc_ids, relevant_doc_ids),
            "retrieved_doc_ids_top": "; ".join(retrieved_doc_ids[:top_k]),
        }

        for k in ks:
            item[f"doc_hit@{k}"] = hit_at_k(retrieved_doc_ids, relevant_doc_ids, k)
            item[f"doc_ndcg@{k}"] = ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, k)

        rows.append(item)

    details = pd.DataFrame(rows)

    metric_cols = [
        c for c in details.columns
        if c.startswith("doc_hit@")
        or c.startswith("doc_ndcg@")
        or c == "doc_mrr"
    ]

    summary = pd.DataFrame([details[metric_cols].mean(numeric_only=True)]).assign(
        method=method,
        top_k=top_k,
        bm25_weight=bm25_weight,
        frida_weight=frida_weight,
        candidate_k=candidate_k,
        k_rrf=k_rrf,
        n_queries=len(details),
    )

    return details, summary


def plot_metric(summary: pd.DataFrame, prefix: str, title: str, ylabel: str, out_path: Path) -> None:
    cols = []

    for col in summary.columns:
        if col.startswith(prefix + "@"):
            k = int(col.split("@")[1])
            cols.append((k, col))

    cols.sort(key=lambda x: x[0])

    if not cols:
        raise ValueError(f"No columns for prefix: {prefix}")

    ks = [k for k, _ in cols]
    values = [float(summary.iloc[0][col]) for _, col in cols]

    plt.figure(figsize=(9, 5))
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


def main() -> None:
    args = parse_args()

    df = load_dataset(args.dataset)

    details, summary = evaluate(
        df=df,
        method=args.method,
        top_k=args.top_k,
        ks=args.ks,
        bm25_weight=args.bm25_weight,
        frida_weight=args.frida_weight,
        candidate_k=args.candidate_k,
        k_rrf=args.k_rrf,
    )

    args.details_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    details.to_csv(args.details_out, index=False)
    summary.to_csv(args.summary_out, index=False)

    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(summary.to_string(index=False))

    print("\nSaved details:", args.details_out)
    print("Saved summary:", args.summary_out)

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


if __name__ == "__main__":
    main()
