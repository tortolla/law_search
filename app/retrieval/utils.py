import re
import numpy as np


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_bm25(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_bm25(text: str):
    return normalize_for_bm25(text).split()


def split_long_piece(piece: str, max_piece_size: int = 1000, overlap: int = 150):
    piece = piece.strip()
    if not piece:
        return []

    if len(piece) <= max_piece_size:
        return [piece]

    sentences = re.split(r"(?<=[\.\!\?])\s+", piece)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [
            piece[i:i + max_piece_size]
            for i in range(0, len(piece), max_piece_size - overlap)
        ]

    subchunks = []
    current = ""

    for sent in sentences:
        if not current:
            current = sent
            continue

        if len(current) + 1 + len(sent) <= max_piece_size:
            current += " " + sent
        else:
            subchunks.append(current.strip())

            tail = current[-overlap:] if overlap > 0 and len(current) > overlap else current
            current = (tail + " " + sent).strip()

            if len(current) > max_piece_size:
                hard_parts = [
                    current[i:i + max_piece_size]
                    for i in range(0, len(current), max_piece_size - overlap)
                ]
                subchunks.extend(hard_parts[:-1])
                current = hard_parts[-1]

    if current.strip():
        subchunks.append(current.strip())

    return subchunks


def split_text_into_chunks(
    text: str,
    chunk_size: int = 1800,
    chunk_overlap: int = 300,
    max_paragraph_size: int = 1000,
    long_piece_overlap: int = 150,
):
    text = clean_text(text)
    if not text:
        return []

    raw_pieces = re.split(r"\n\s*\n", text)
    raw_pieces = [p.strip() for p in raw_pieces if p.strip()]

    pieces = []
    for piece in raw_pieces:
        pieces.extend(
            split_long_piece(
                piece,
                max_piece_size=max_paragraph_size,
                overlap=long_piece_overlap,
            )
        )

    chunks = []
    current = ""

    for piece in pieces:
        if not current:
            current = piece
            continue

        candidate = current + "\n\n" + piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current.strip())

            tail = current[-chunk_overlap:] if chunk_overlap > 0 and len(current) > chunk_overlap else current
            current = (tail + "\n\n" + piece).strip()

            while len(current) > chunk_size:
                chunks.append(current[:chunk_size].strip())
                current = current[max(1, chunk_size - chunk_overlap):].strip()

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c]


def minmax_norm(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) == 0:
        return arr
    mn = arr.min()
    mx = arr.max()
    if abs(mx - mn) < 1e-12:
        return np.ones_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)