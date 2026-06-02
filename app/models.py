from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)


class TextResponse(BaseModel):
    ok: bool = True
    result: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=20000)
    top_k: int = Field(default=20, ge=1, le=60)
    search_mode: Literal["hybrid", "weighted", "rrf"] = "hybrid"


class SearchItem(BaseModel):
    rank: int
    doc_id: str
    title: Optional[str] = None
    category: Optional[str] = None
    chunk_id: str
    chunk_ix: int

    hybrid_score: Optional[float] = None
    bm25_score: Optional[float] = None
    frida_score: Optional[float] = None
    rrf_score: Optional[float] = None

    txt_path: Optional[str] = None
    meta_file: Optional[str] = None
    chunk_text: str
    method: Optional[str] = None

    dense_backend: Optional[str] = None
    collection: Optional[str] = None


class SearchResponse(BaseModel):
    ok: bool = True
    query: str
    top_k: int
    search_mode: str
    vector_backend: Optional[str] = None
    items: List[SearchItem]


class ExpandContextRequest(BaseModel):
    doc_id: str = Field(..., min_length=1)
    chunk_ix: int = Field(..., ge=0)
    window_before: int = Field(default=2, ge=0, le=10)
    window_after: int = Field(default=2, ge=0, le=10)


class ContextItem(BaseModel):
    chunk_ix: int
    chunk_id: str
    chunk_text: str


class ExpandContextResponse(BaseModel):
    ok: bool = True
    doc_id: str
    center_chunk_ix: int
    window_before: int
    window_after: int
    items: List[ContextItem]
    merged_text: str
