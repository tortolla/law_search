from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.auth import verify_api_key, verify_ingest_key, verify_result_key
from app.models import (
    ExpandContextRequest,
    ExpandContextResponse,
    SearchRequest,
    SearchResponse,
    TextRequest,
    TextResponse,
)
from app.processor import process_expand_context, process_search, process_text

load_dotenv()

app = FastAPI(title="Local Dify Bridge", version="0.2.0")


@app.get("/health")
def health():
    return {"ok": True, "service": "local_dify_bridge"}


# legacy endpoints
@app.post("/ingest", response_model=TextResponse, dependencies=[Depends(verify_ingest_key)])
def ingest_text(payload: TextRequest):
    result = process_text(payload.text)
    return TextResponse(result=result)


@app.post("/result", response_model=TextResponse, dependencies=[Depends(verify_result_key)])
def result_text(payload: TextRequest):
    result = process_text(payload.text)
    return TextResponse(result=result)


# new structured endpoints for Dify tools
@app.post(
    "/search_base_articles",
    response_model=SearchResponse,
    dependencies=[Depends(verify_api_key)],
)
def search_base_articles(payload: SearchRequest):
    return process_search(
        query=payload.query,
        top_k=payload.top_k,
        search_mode=payload.search_mode,
    )


@app.post(
    "/expand_context_chunks",
    response_model=ExpandContextResponse,
    dependencies=[Depends(verify_api_key)],
)
def expand_context_chunks(payload: ExpandContextRequest):
    return process_expand_context(
        doc_id=payload.doc_id,
        chunk_ix=payload.chunk_ix,
        window_before=payload.window_before,
        window_after=payload.window_after,
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={
            "ok": False,
            "error": "bad_request",
            "detail": str(exc),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "error": "internal_server_error",
            "detail": str(exc),
        },
    )