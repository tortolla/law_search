import os
from fastapi import Header, HTTPException


def _check_key(received_key: str | None, expected_key: str, scope: str) -> None:
    if not expected_key:
        raise HTTPException(status_code=500, detail=f"{scope}_api_key_not_configured")

    if received_key != expected_key:
        raise HTTPException(status_code=401, detail=f"{scope}_unauthorized")


def verify_ingest_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("DIFY_INGEST_API_KEY", "")
    _check_key(x_api_key, expected, "ingest")


def verify_result_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("DIFY_RESULT_API_KEY", "")
    _check_key(x_api_key, expected, "result")


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = (
        os.getenv("DIFY_API_KEY")
        or os.getenv("DIFY_INGEST_API_KEY")
        or os.getenv("DIFY_RESULT_API_KEY")
        or ""
    )
    _check_key(x_api_key, expected, "api")
