import numpy as np

from app.retrieval.config import (
    DEFAULT_TOP_K,
    DEFAULT_CANDIDATE_K,
    DEFAULT_BM25_WEIGHT,
    DEFAULT_FRIDA_WEIGHT,
)
from app.retrieval.runtime import load_artifacts
from app.retrieval.utils import tokenize_bm25, minmax_norm




def _safe_l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return x / norms


def _safe_l2_normalize_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    norm = np.linalg.norm(x)
    if norm == 0.0:
        norm = 1.0
    return x / norm


def cosine_scores_for_query(query: str, encoder, doc_embeddings: np.ndarray) -> np.ndarray:
    q_emb = encoder.encode(
        query,
        prompt_name="search_query",
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    q_emb = _safe_l2_normalize_vec(q_emb)

    doc_embeddings = _safe_l2_normalize_rows(doc_embeddings)

    scores = np.einsum("ij,j->i", doc_embeddings, q_emb, optimize=True)
    scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
    return scores.astype(np.float32)


def search_frida(query: str, top_k: int = DEFAULT_TOP_K):
    artifacts = load_artifacts()
    scores = cosine_scores_for_query(query, artifacts.frida_encoder, artifacts.frida_embeddings)
    top_idx = np.argsort(scores)[::-1][:top_k]

    res = artifacts.chunks_df.iloc[top_idx].copy()
    res["frida_score"] = scores[top_idx]
    res["method"] = "frida"

    return res[
        [
            "doc_id",
            "title",
            "category",
            "chunk_id",
            "chunk_ix",
            "frida_score",
            "txt_path",
            "meta_file",
            "chunk_text",
        ]
    ].reset_index(drop=True)

def _ranks_from_scores_desc(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores)[::-1]
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks

def search_bm25_frida(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    frida_weight: float = DEFAULT_FRIDA_WEIGHT,
    candidate_k: int = DEFAULT_CANDIDATE_K,
):
    artifacts = load_artifacts()

    q_tokens = tokenize_bm25(query)
    bm25_scores = np.array(artifacts.bm25.get_scores(q_tokens), dtype=np.float32)

    frida_scores = cosine_scores_for_query(
        query,
        artifacts.frida_encoder,
        artifacts.frida_embeddings,
    ).astype(np.float32)

    top_bm25_idx = np.argsort(bm25_scores)[::-1][:candidate_k]
    top_frida_idx = np.argsort(frida_scores)[::-1][:candidate_k]
    candidate_idx = np.unique(np.concatenate([top_bm25_idx, top_frida_idx]))

    cand_bm25 = bm25_scores[candidate_idx]
    cand_frida = frida_scores[candidate_idx]

    bm25_norm = minmax_norm(cand_bm25)
    frida_norm = minmax_norm(cand_frida)

    hybrid_scores = bm25_weight * bm25_norm + frida_weight * frida_norm
    hybrid_scores = np.nan_to_num(hybrid_scores, nan=-1.0, posinf=-1.0, neginf=-1.0)

    order = np.argsort(hybrid_scores)[::-1][:top_k]
    final_idx = candidate_idx[order]

    res = artifacts.chunks_df.iloc[final_idx].copy()
    res["hybrid_score"] = hybrid_scores[order]
    res["bm25_score"] = cand_bm25[order]
    res["frida_score"] = cand_frida[order]
    res["method"] = "bm25+frida"

    return res[
        [
            "doc_id",
            "title",
            "category",
            "chunk_id",
            "chunk_ix",
            "hybrid_score",
            "bm25_score",
            "frida_score",
            "txt_path",
            "meta_file",
            "chunk_text",
        ]
    ].reset_index(drop=True)

def search_bm25_frida_weighted(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    frida_weight: float = DEFAULT_FRIDA_WEIGHT,
    candidate_k: int = DEFAULT_CANDIDATE_K,
):
    artifacts = load_artifacts()

    q_tokens = tokenize_bm25(query)
    bm25_scores = np.array(artifacts.bm25.get_scores(q_tokens), dtype=np.float32)

    frida_scores = cosine_scores_for_query(
        query,
        artifacts.frida_encoder,
        artifacts.frida_embeddings,
    ).astype(np.float32)

    top_bm25_idx = np.argsort(bm25_scores)[::-1][:candidate_k]
    top_frida_idx = np.argsort(frida_scores)[::-1][:candidate_k]
    candidate_idx = np.unique(np.concatenate([top_bm25_idx, top_frida_idx]))

    cand_bm25 = bm25_scores[candidate_idx]
    cand_frida = frida_scores[candidate_idx]

    bm25_norm = minmax_norm(cand_bm25)
    frida_norm = minmax_norm(cand_frida)

    hybrid_scores = bm25_weight * bm25_norm + frida_weight * frida_norm
    hybrid_scores = np.nan_to_num(hybrid_scores, nan=-1.0, posinf=-1.0, neginf=-1.0)

    order = np.argsort(hybrid_scores)[::-1][:top_k]
    final_idx = candidate_idx[order]

    res = artifacts.chunks_df.iloc[final_idx].copy()
    res["hybrid_score"] = hybrid_scores[order]
    res["bm25_score"] = cand_bm25[order]
    res["frida_score"] = cand_frida[order]
    res["method"] = "bm25+frida_weighted"

    return res[
        [
            "doc_id",
            "title",
            "category",
            "chunk_id",
            "chunk_ix",
            "hybrid_score",
            "bm25_score",
            "frida_score",
            "txt_path",
            "meta_file",
            "chunk_text",
        ]
    ].reset_index(drop=True)


def search_bm25_frida_rrf(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    bm25_weight: float = 1.0,
    frida_weight: float = 1.0,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    k_rrf: int = 60,
):
    artifacts = load_artifacts()

    q_tokens = tokenize_bm25(query)
    bm25_scores = np.array(artifacts.bm25.get_scores(q_tokens), dtype=np.float32)

    frida_scores = cosine_scores_for_query(
        query,
        artifacts.frida_encoder,
        artifacts.frida_embeddings,
    ).astype(np.float32)

    top_bm25_idx = np.argsort(bm25_scores)[::-1][:candidate_k]
    top_frida_idx = np.argsort(frida_scores)[::-1][:candidate_k]
    candidate_idx = np.unique(np.concatenate([top_bm25_idx, top_frida_idx]))

    cand_bm25 = bm25_scores[candidate_idx]
    cand_frida = frida_scores[candidate_idx]

    bm25_ranks = _ranks_from_scores_desc(cand_bm25)
    frida_ranks = _ranks_from_scores_desc(cand_frida)

    rrf_scores = (
        bm25_weight / (k_rrf + bm25_ranks.astype(np.float32))
        + frida_weight / (k_rrf + frida_ranks.astype(np.float32))
    )
    rrf_scores = np.nan_to_num(rrf_scores, nan=-1.0, posinf=-1.0, neginf=-1.0)

    order = np.argsort(rrf_scores)[::-1][:top_k]
    final_idx = candidate_idx[order]

    res = artifacts.chunks_df.iloc[final_idx].copy()
    res["rrf_score"] = rrf_scores[order]
    res["bm25_score"] = cand_bm25[order]
    res["frida_score"] = cand_frida[order]
    res["method"] = "bm25+frida_rrf"

    return res[
        [
            "doc_id",
            "title",
            "category",
            "chunk_id",
            "chunk_ix",
            "rrf_score",
            "bm25_score",
            "frida_score",
            "txt_path",
            "meta_file",
            "chunk_text",
        ]
    ].reset_index(drop=True)

def search_dispatch(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    search_mode: str = "hybrid",
):
    if search_mode == "hybrid":
        return search_bm25_frida(query=query, top_k=top_k)
    if search_mode == "weighted":
        return search_bm25_frida_weighted(query=query, top_k=top_k)
    if search_mode == "rrf":
        return search_bm25_frida_rrf(query=query, top_k=top_k)

    raise ValueError(f"Unknown search_mode: {search_mode}")


def _doc_first_unique_preserve_order(items):
    seen = set()
    out = []

    for item in items:
        item = str(item)
        if item not in seen:
            seen.add(item)
            out.append(item)

    return out


def _doc_first_base_search(
    query: str,
    base_search_mode: str,
    retrieval_top_k: int,
    bm25_weight: float,
    frida_weight: float,
    candidate_k: int,
    k_rrf: int,
):
    if base_search_mode == "hybrid":
        return search_bm25_frida(
            query=query,
            top_k=retrieval_top_k,
            bm25_weight=bm25_weight,
            frida_weight=frida_weight,
            candidate_k=candidate_k,
        )

    if base_search_mode == "weighted":
        return search_bm25_frida_weighted(
            query=query,
            top_k=retrieval_top_k,
            bm25_weight=bm25_weight,
            frida_weight=frida_weight,
            candidate_k=candidate_k,
        )

    if base_search_mode == "rrf":
        return search_bm25_frida_rrf(
            query=query,
            top_k=retrieval_top_k,
            bm25_weight=bm25_weight,
            frida_weight=frida_weight,
            candidate_k=candidate_k,
            k_rrf=k_rrf,
        )

    if base_search_mode == "frida":
        return search_frida(
            query=query,
            top_k=retrieval_top_k,
        )

    raise ValueError(f"Unknown base_search_mode: {base_search_mode}")


def search_doc_first_top_chunks(
    query: str,
    doc_top_k: int = 30,
    chunks_per_doc: int = 3,
    retrieval_top_k: int = 300,
    base_search_mode: str = "weighted",
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    frida_weight: float = DEFAULT_FRIDA_WEIGHT,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    k_rrf: int = 60,
):
    """
    Новый отдельный режим поиска.

    1. Старым поиском ищем retrieval_top_k чанков.
    2. Из них берем doc_top_k уникальных документов.
    3. Внутри каждого документа выбираем chunks_per_doc чанков
       по cosine similarity к запросу.
    4. Возвращаем итоговые чанки.

    Старые функции не меняются.
    """

    artifacts = load_artifacts()

    base_results = _doc_first_base_search(
        query=query,
        base_search_mode=base_search_mode,
        retrieval_top_k=retrieval_top_k,
        bm25_weight=bm25_weight,
        frida_weight=frida_weight,
        candidate_k=candidate_k,
        k_rrf=k_rrf,
    )

    if base_results.empty:
        return base_results

    top_doc_ids = _doc_first_unique_preserve_order(
        base_results["doc_id"].astype(str).tolist()
    )[:doc_top_k]

    chunks_df = artifacts.chunks_df.copy()
    chunks_df["doc_id"] = chunks_df["doc_id"].astype(str)
    chunks_df["chunk_id"] = chunks_df["chunk_id"].astype(str)
    chunks_df["chunk_ix"] = chunks_df["chunk_ix"].astype(int)

    frida_scores = cosine_scores_for_query(
        query,
        artifacts.frida_encoder,
        artifacts.frida_embeddings,
    ).astype(np.float32)

    parts = []

    for doc_rank, doc_id in enumerate(top_doc_ids, start=1):
        doc_chunks = chunks_df[chunks_df["doc_id"] == doc_id].copy()

        if doc_chunks.empty:
            continue

        doc_indexes = doc_chunks.index.to_numpy()
        doc_chunks["chunk_cosine_score"] = frida_scores[doc_indexes]
        doc_chunks["doc_rank"] = doc_rank

        doc_chunks = (
            doc_chunks
            .sort_values(
                ["chunk_cosine_score", "chunk_ix"],
                ascending=[False, True],
            )
            .head(chunks_per_doc)
            .copy()
        )

        doc_chunks["chunk_rank_in_doc"] = range(1, len(doc_chunks) + 1)
        parts.append(doc_chunks)

    if not parts:
        return chunks_df.iloc[0:0].copy()

    result = parts[0]
    if len(parts) > 1:
        result = result._append(parts[1:], ignore_index=True)

    result = (
        result
        .sort_values(["doc_rank", "chunk_rank_in_doc"], ascending=[True, True])
        .reset_index(drop=True)
    )

    result["rank"] = range(1, len(result) + 1)
    result["method"] = "doc_first_top_chunks"
    result["doc_first_score"] = 1.0 / result["doc_rank"].astype(float)

    base_doc_meta = base_results.drop_duplicates(subset=["doc_id"], keep="first").copy()
    base_doc_meta["doc_id"] = base_doc_meta["doc_id"].astype(str)

    meta_cols = ["doc_id"]
    for col in ["hybrid_score", "rrf_score", "bm25_score", "frida_score"]:
        if col in base_doc_meta.columns:
            meta_cols.append(col)

    base_doc_meta = base_doc_meta[meta_cols]

    result = result.merge(
        base_doc_meta,
        on="doc_id",
        how="left",
        suffixes=("", "_base"),
    )

    output_cols = [
        "rank",
        "doc_rank",
        "chunk_rank_in_doc",
        "doc_id",
        "title",
        "category",
        "chunk_id",
        "chunk_ix",
        "chunk_cosine_score",
        "doc_first_score",
        "hybrid_score",
        "rrf_score",
        "bm25_score",
        "frida_score",
        "txt_path",
        "meta_file",
        "chunk_text",
        "method",
    ]

    output_cols = [col for col in output_cols if col in result.columns]

    return result[output_cols].reset_index(drop=True)