from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

from app.retrieval.config import get_gold_dataset_path, REPORTS_DIR
from app.retrieval.search import (
    search_frida,
    search_bm25_frida_weighted,
    search_bm25_frida_rrf,
    search_doc_first_top_chunks,
)


DEFAULT_KS = [1, 5, 10, 15, 20, 25, 35, 50, 70, 90, 120, 150]


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def fail(message: str) -> None:
    print(f"[ERROR] {message}")
    sys.exit(1)


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    out: list[str] = []

    for x in items:
        x = str(x)
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


def first_hit_rank(ranked: Sequence[str], relevant: set[str]) -> int | None:
    for i, item in enumerate(ranked, start=1):
        if item in relevant:
            return i
    return None


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


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"dataset not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        fail(f"cannot read dataset json: {path}; error={e}")

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
        fail(f"missing required columns in dataset: {missing}")

    return df


def inspect_gold_coverage(df: pd.DataFrame) -> None:
    print()
    print("DATASET")
    print("-" * 100)

    print(f"queries: {len(df)}")
    print(f"unique source_doc_id: {df['source_doc_id'].astype(str).nunique()}")
    print(f"unique source_chunk_id: {df['source_chunk_id'].astype(str).nunique()}")

    print()
    print("query_type:")
    print(df["query_type"].value_counts(dropna=False).to_string())

    print()
    print("difficulty:")
    print(df["difficulty"].value_counts(dropna=False).to_string())

    print()
    print("category:")
    print(df["category"].value_counts(dropna=False).to_string())

    warn(
        "Current dataset is valid mainly for document-level evaluation. "
        "Chunk-level metrics may be invalid after rechunking if old gold chunk_id values "
        "are absent from current chunks.parquet."
    )


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

    raise ValueError(f"unknown method: {method}")


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ks = sorted(set(int(k) for k in ks if 0 < int(k) <= top_k))

    if not ks:
        fail("no valid K values after filtering by top_k")

    rows = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"Eval {method}"):
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

        if "chunk_id" not in results.columns or "doc_id" not in results.columns:
            fail("search results must contain chunk_id and doc_id columns")

        retrieved_chunk_ids = results["chunk_id"].astype(str).tolist()
        retrieved_doc_ids = unique_preserve_order(results["doc_id"].astype(str).tolist())

        relevant_chunk_ids = set(map(str, row.relevant_chunk_ids))
        relevant_doc_ids = set(map(str, row.relevant_doc_ids))

        first_doc_rank = first_hit_rank(retrieved_doc_ids, relevant_doc_ids)
        first_chunk_rank = first_hit_rank(retrieved_chunk_ids, relevant_chunk_ids)

        item = {
            "query": row.query,
            "query_type": row.query_type,
            "difficulty": row.difficulty,
            "category": row.category,
            "source_doc_id": str(row.source_doc_id),
            "source_chunk_id": str(row.source_chunk_id),
            "first_doc_rank": first_doc_rank,
            "first_chunk_rank": first_chunk_rank,
            "doc_mrr": reciprocal_rank(retrieved_doc_ids, relevant_doc_ids),
            "chunk_mrr": reciprocal_rank(retrieved_chunk_ids, relevant_chunk_ids),
            "retrieved_doc_ids_top10": json.dumps(retrieved_doc_ids[:10], ensure_ascii=False),
            "retrieved_chunk_ids_top10": json.dumps(retrieved_chunk_ids[:10], ensure_ascii=False),
        }

        for k in ks:
            item[f"doc_hit@{k}"] = hit_at_k(retrieved_doc_ids, relevant_doc_ids, k)
            item[f"doc_error@{k}"] = 1.0 - item[f"doc_hit@{k}"]
            item[f"doc_ndcg@{k}"] = ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, k)

            item[f"chunk_hit@{k}"] = hit_at_k(retrieved_chunk_ids, relevant_chunk_ids, k)
            item[f"chunk_error@{k}"] = 1.0 - item[f"chunk_hit@{k}"]
            item[f"chunk_ndcg@{k}"] = ndcg_at_k(retrieved_chunk_ids, relevant_chunk_ids, k)

        rows.append(item)

    details = pd.DataFrame(rows)

    metric_cols = [
        c for c in details.columns
        if c.endswith("_mrr")
        or "_hit@" in c
        or "_error@" in c
        or "_ndcg@" in c
    ]

    overall = pd.DataFrame([details[metric_cols].mean(numeric_only=True)]).assign(
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
        n_queries=len(details),
        created_at=datetime.now().isoformat(timespec="seconds"),
    )

    by_query_type = (
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

    curve_rows = []
    for k in ks:
        hit_col = f"doc_hit@{k}"
        error_col = f"doc_error@{k}"
        ndcg_col = f"doc_ndcg@{k}"

        curve_rows.append(
            {
                "documents_number": k,
                "doc_accuracy": float(overall.iloc[0][hit_col]),
                "doc_error_rate": float(overall.iloc[0][error_col]),
                "doc_accuracy_percent": 100.0 * float(overall.iloc[0][hit_col]),
                "doc_error_percent": 100.0 * float(overall.iloc[0][error_col]),
                "doc_ndcg": float(overall.iloc[0][ndcg_col]),
                "n_queries": len(details),
            }
        )

    doc_curve = pd.DataFrame(curve_rows)

    return details, overall, by_query_type, by_difficulty, by_category, doc_curve


def plot_doc_curves(doc_curve: pd.DataFrame, out_dir: Path) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    x = doc_curve["documents_number"]

    plt.figure(figsize=(13, 6))
    plt.plot(x, doc_curve["doc_accuracy"], marker="o", linewidth=2)
    plt.title("Document Accuracy Dependence on the Number of Retrieved Documents")
    plt.xlabel("Documents Number / Top-K")
    plt.ylabel("Accuracy, doc_hit@K")
    plt.ylim(0, 1.03)
    plt.xticks(x)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = plots_dir / "doc_accuracy_curve.png"
    plt.savefig(out_path, dpi=220)
    plt.close()
    ok(f"saved plot: {out_path}")

    plt.figure(figsize=(13, 6))
    plt.plot(x, doc_curve["doc_error_percent"], marker="o", linewidth=2)
    plt.title("Document Error Rate Dependence on the Number of Retrieved Documents")
    plt.xlabel("Documents Number / Top-K")
    plt.ylabel("Error rate, %")
    plt.ylim(0, 100)
    plt.xticks(x)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = plots_dir / "doc_error_curve.png"
    plt.savefig(out_path, dpi=220)
    plt.close()
    ok(f"saved plot: {out_path}")


def parse_args() -> argparse.Namespace:
    default_dataset = get_gold_dataset_path()

    parser = argparse.ArgumentParser(
        description="Evaluate retrieval. Default mode is document-level weighted hybrid evaluation."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset if default_dataset is not None else Path("data/processed/dataset_fixed.json"),
        help="Path to eval dataset JSON.",
    )

    parser.add_argument(
        "--method",
        choices=["frida", "weighted", "rrf", "doc_first"],
        default="weighted",
        help="Retrieval method. Default: weighted.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=150,
        help="Max retrieval depth. Default: 150.",
    )

    parser.add_argument(
        "--ks",
        type=int,
        nargs="*",
        default=DEFAULT_KS,
        help="Cutoffs to report. Default: 1 5 10 15 20 25 35 50 70 90 120 150.",
    )

    parser.add_argument("--bm25-weight", type=float, default=0.3)
    parser.add_argument("--frida-weight", type=float, default=0.7)
    parser.add_argument("--candidate-k", type=int, default=1000)
    parser.add_argument("--k-rrf", type=int, default=60)

    parser.add_argument("--doc-top-k", type=int, default=30)
    parser.add_argument("--chunks-per-doc", type=int, default=3)
    parser.add_argument("--retrieval-top-k", type=int, default=300)
    parser.add_argument(
        "--base-search-mode",
        choices=["frida", "weighted", "rrf"],
        default="weighted",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPORTS_DIR / "eval" / "weighted_doc_level",
        help="Output directory for CSV and plots.",
    )

    return parser.parse_args()


def save_outputs(
    out_dir: Path,
    args: argparse.Namespace,
    details: pd.DataFrame,
    overall: pd.DataFrame,
    by_query_type: pd.DataFrame,
    by_difficulty: pd.DataFrame,
    by_category: pd.DataFrame,
    doc_curve: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    details_path = out_dir / "details.csv"
    overall_path = out_dir / "overall.csv"
    by_query_type_path = out_dir / "by_query_type.csv"
    by_difficulty_path = out_dir / "by_difficulty.csv"
    by_category_path = out_dir / "by_category.csv"
    doc_curve_path = out_dir / "doc_curve.csv"
    config_path = out_dir / "run_config.json"

    details.to_csv(details_path, index=False)
    overall.to_csv(overall_path, index=False)
    by_query_type.to_csv(by_query_type_path, index=False)
    by_difficulty.to_csv(by_difficulty_path, index=False)
    by_category.to_csv(by_category_path, index=False)
    doc_curve.to_csv(doc_curve_path, index=False)

    config = {
        "dataset": str(args.dataset),
        "method": args.method,
        "top_k": args.top_k,
        "ks": args.ks,
        "bm25_weight": args.bm25_weight,
        "frida_weight": args.frida_weight,
        "candidate_k": args.candidate_k,
        "k_rrf": args.k_rrf,
        "doc_top_k": args.doc_top_k,
        "chunks_per_doc": args.chunks_per_doc,
        "retrieval_top_k": args.retrieval_top_k,
        "base_search_mode": args.base_search_mode,
        "out_dir": str(out_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ok(f"saved details:       {details_path}")
    ok(f"saved overall:       {overall_path}")
    ok(f"saved by_query_type: {by_query_type_path}")
    ok(f"saved by_difficulty: {by_difficulty_path}")
    ok(f"saved by_category:   {by_category_path}")
    ok(f"saved doc_curve:     {doc_curve_path}")
    ok(f"saved run_config:    {config_path}")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()

    print("=" * 100)
    print("EVALUATE RETRIEVAL")
    print("=" * 100)
    print(f"dataset:        {args.dataset}")
    print(f"method:         {args.method}")
    print(f"top_k:          {args.top_k}")
    print(f"ks:             {args.ks}")
    print(f"bm25_weight:    {args.bm25_weight}")
    print(f"frida_weight:   {args.frida_weight}")
    print(f"candidate_k:    {args.candidate_k}")
    print(f"out_dir:        {out_dir}")

    df = load_dataset(args.dataset)
    inspect_gold_coverage(df)

    details, overall, by_query_type, by_difficulty, by_category, doc_curve = evaluate_dataset(
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

    print()
    print("=" * 100)
    print("OVERALL")
    print("=" * 100)
    print(overall.to_string(index=False))

    print()
    print("=" * 100)
    print("DOC CURVE")
    print("=" * 100)
    print(doc_curve.to_string(index=False))

    print()
    print("=" * 100)
    print("BY CATEGORY")
    print("=" * 100)
    print(by_category.to_string(index=False))

    save_outputs(
        out_dir=out_dir,
        args=args,
        details=details,
        overall=overall,
        by_query_type=by_query_type,
        by_difficulty=by_difficulty,
        by_category=by_category,
        doc_curve=doc_curve,
    )

    plot_doc_curves(doc_curve, out_dir)

    print()
    print("=" * 100)
    ok("retrieval evaluation completed")
    print("=" * 100)


if __name__ == "__main__":
    main()
