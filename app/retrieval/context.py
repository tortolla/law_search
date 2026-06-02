from app.retrieval.runtime import load_artifacts


def expand_context_chunks(
    doc_id: str,
    chunk_ix: int,
    window_before: int = 2,
    window_after: int = 2,
) -> dict:
    artifacts = load_artifacts()
    df = artifacts.chunks_df

    doc_df = df[df["doc_id"] == doc_id].copy()
    if doc_df.empty:
        raise ValueError(f"doc_id not found: {doc_id}")

    doc_df = doc_df.sort_values("chunk_ix").reset_index(drop=True)

    left_ix = max(0, chunk_ix - window_before)
    right_ix = chunk_ix + window_after

    context_df = doc_df[
        (doc_df["chunk_ix"] >= left_ix) & (doc_df["chunk_ix"] <= right_ix)
    ].copy()

    if context_df.empty:
        raise ValueError(
            f"context chunks not found for doc_id={doc_id}, chunk_ix={chunk_ix}"
        )

    items = []
    for _, row in context_df.iterrows():
        items.append(
            {
                "chunk_ix": int(row["chunk_ix"]),
                "chunk_id": str(row["chunk_id"]),
                "chunk_text": str(row["chunk_text"]),
            }
        )

    merged_text = "\n\n".join(
        [
            f"[chunk_ix={item['chunk_ix']}, chunk_id={item['chunk_id']}]\n{item['chunk_text']}"
            for item in items
        ]
    )

    return {
        "ok": True,
        "doc_id": str(doc_id),
        "center_chunk_ix": int(chunk_ix),
        "window_before": int(window_before),
        "window_after": int(window_after),
        "items": items,
        "merged_text": merged_text,
    }