#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

export PYTHONPATH="${PYTHONPATH:-.}"

# API auth for local testing
export DIFY_API_KEY="${DIFY_API_KEY:-local-dev-key}"
export DIFY_INGEST_API_KEY="${DIFY_INGEST_API_KEY:-local-dev-key}"
export DIFY_RESULT_API_KEY="${DIFY_RESULT_API_KEY:-local-dev-key}"

# Retrieval backend
export VECTOR_BACKEND="${VECTOR_BACKEND:-milvus}"

# Local Milvus from Docker Compose
export MILVUS_HOST="${MILVUS_HOST:-localhost}"
export MILVUS_PORT="${MILVUS_PORT:-19530}"

# Local debug collection.
# For production/full base change this in .env:
# MILVUS_COLLECTION=frida_chunks
export MILVUS_COLLECTION="${MILVUS_COLLECTION:-frida_chunks_test_50}"

echo "===================================================================================================="
echo "RUN LOCAL API"
echo "===================================================================================================="
echo "PROJECT_ROOT:       $(pwd)"
echo "PYTHONPATH:         $PYTHONPATH"
echo "DIFY_API_KEY:       $DIFY_API_KEY"
echo "VECTOR_BACKEND:     $VECTOR_BACKEND"
echo "MILVUS_HOST:        $MILVUS_HOST"
echo "MILVUS_PORT:        $MILVUS_PORT"
echo "MILVUS_COLLECTION:  $MILVUS_COLLECTION"
echo "===================================================================================================="

if [[ ! -x ".venv/bin/python" ]]; then
  echo "[ERROR] .venv/bin/python not found"
  echo "Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
