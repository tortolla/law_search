#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
else
  echo "[WARN] .env not found. Creating from .env.example"
  cp .env.example .env
  set -a
  source ".env"
  set +a
fi

export PYTHONPATH="${PYTHONPATH:-.}"

RAW_DATA_DIR="${RAW_DATA_DIR:-data_big}"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed}"
INDEXES_DIR="${INDEXES_DIR:-data/indexes}"
MILVUS_HOST="${MILVUS_HOST:-localhost}"
MILVUS_PORT="${MILVUS_PORT:-19530}"
MILVUS_COLLECTION="${MILVUS_COLLECTION:-frida_chunks}"
GOLD_DATASET_PATH="${GOLD_DATASET_PATH:-data/processed/dataset_fixed.json}"

FORCE=0
SKIP_DOWNLOAD=0
SKIP_EMBEDDINGS=0
SKIP_EVAL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    --skip-download)
      SKIP_DOWNLOAD=1
      shift
      ;;
    --skip-embeddings)
      SKIP_EMBEDDINGS=1
      shift
      ;;
    --skip-eval)
      SKIP_EVAL=1
      shift
      ;;
    *)
      echo "[ERROR] unknown argument: $1"
      exit 1
      ;;
  esac
done

run_step() {
  echo
  echo "===================================================================================================="
  echo "$1"
  echo "===================================================================================================="
  shift
  "$@"
}

file_exists_nonempty() {
  [[ -f "$1" && -s "$1" ]]
}

dir_exists_nonempty() {
  [[ -d "$1" && -n "$(ls -A "$1" 2>/dev/null || true)" ]]
}

wait_for_milvus() {
  echo "[INFO] waiting for Milvus at ${MILVUS_HOST}:${MILVUS_PORT}"

  for i in $(seq 1 90); do
    if .venv/bin/python - <<PY >/dev/null 2>&1
from pymilvus import connections, utility
connections.connect(alias="default", host="${MILVUS_HOST}", port="${MILVUS_PORT}")
utility.list_collections()
PY
    then
      echo "[OK] Milvus is ready"
      return 0
    fi

    echo "[WAIT] Milvus not ready: ${i}/90"
    sleep 3
  done

  echo "[ERROR] Milvus did not become ready"
  exit 1
}

echo "===================================================================================================="
echo "FULL SETUP: LOCAL DIFY BRIDGE + MILVUS"
echo "===================================================================================================="
echo "RAW_DATA_DIR:       $RAW_DATA_DIR"
echo "PROCESSED_DIR:      $PROCESSED_DIR"
echo "INDEXES_DIR:        $INDEXES_DIR"
echo "FRIDA_MODEL_ID:     ${FRIDA_MODEL_ID:-ai-forever/FRIDA}"
echo "FRIDA_MODEL_PATH:   ${FRIDA_MODEL_PATH:-models/FRIDA}"
echo "PUBLIC_DATA_URL:    ${PUBLIC_DATA_URL:-}"
echo "MILVUS_HOST:        $MILVUS_HOST"
echo "MILVUS_PORT:        $MILVUS_PORT"
echo "MILVUS_COLLECTION:  $MILVUS_COLLECTION"
echo "FORCE:              $FORCE"
echo "SKIP_DOWNLOAD:      $SKIP_DOWNLOAD"
echo "SKIP_EMBEDDINGS:    $SKIP_EMBEDDINGS"
echo "SKIP_EVAL:          $SKIP_EVAL"
echo "===================================================================================================="

if [[ ! -x ".venv/bin/python" ]]; then
  echo "[INFO] creating virtual environment"
  python3 -m venv .venv
fi

run_step "INSTALL REQUIREMENTS" \
  .venv/bin/pip install -r requirements.txt

run_step "00 DOWNLOAD / CHECK FRIDA MODEL" \
  .venv/bin/python scripts/00_download_frida_model.py

run_step "00 CHECK ENV" \
  .venv/bin/python scripts/00_check_env.py

run_step "START MILVUS" \
  docker-compose up -d etcd minio milvus-standalone

wait_for_milvus

if [[ "$SKIP_DOWNLOAD" == "0" ]]; then
  if [[ "$FORCE" == "1" || ! dir_exists_nonempty "$RAW_DATA_DIR" ]]; then
    run_step "01 DOWNLOAD DATA" \
      .venv/bin/python scripts/01_download_data.py
  else
    echo "[SKIP] raw data already exists: $RAW_DATA_DIR"
  fi
else
  echo "[SKIP] download disabled"
fi

run_step "02 VALIDATE RAW DATA" \
  .venv/bin/python scripts/02_validate_raw_data.py

if [[ "$FORCE" == "1" || ! file_exists_nonempty "$PROCESSED_DIR/chunks.parquet" ]]; then
  run_step "03 BUILD CHUNKS" \
    .venv/bin/python scripts/03_build_chunks.py
else
  echo "[SKIP] chunks already exist: $PROCESSED_DIR/chunks.parquet"
fi

if [[ "$FORCE" == "1" || ! file_exists_nonempty "$INDEXES_DIR/bm25/bm25.pkl" ]]; then
  run_step "04 BUILD BM25" \
    .venv/bin/python scripts/04_build_bm25.py
else
  echo "[SKIP] BM25 already exists: $INDEXES_DIR/bm25/bm25.pkl"
fi

if [[ "$SKIP_EMBEDDINGS" == "0" ]]; then
  if [[ "$FORCE" == "1" || ! file_exists_nonempty "$INDEXES_DIR/frida/embeddings.npy" ]]; then
    run_step "05 BUILD FRIDA EMBEDDINGS" \
      .venv/bin/python scripts/05_build_frida_embeddings.py --device auto
  else
    echo "[SKIP] embeddings already exist: $INDEXES_DIR/frida/embeddings.npy"
  fi
else
  echo "[SKIP] embeddings disabled"
fi

run_step "06 LOAD MILVUS" \
  .venv/bin/python scripts/06_load_milvus.py \
    --mode server \
    --host "$MILVUS_HOST" \
    --port "$MILVUS_PORT" \
    --collection "$MILVUS_COLLECTION" \
    --drop-existing \
    --batch-size 5000

run_step "07 VALIDATE INDEXES" \
  .venv/bin/python scripts/07_validate_indexes.py

if [[ "$SKIP_EVAL" == "0" ]]; then
  if [[ -f "$GOLD_DATASET_PATH" ]]; then
    run_step "08 EVAL SEARCH" \
      .venv/bin/python scripts/08_eval_search.py \
        --dataset "$GOLD_DATASET_PATH" \
        --method weighted \
        --bm25-weight 0.3 \
        --frida-weight 0.7 \
        --candidate-k 1000 \
        --top-k 150 \
        --ks 1 5 10 15 20 25 35 50 70 90 120 150
  else
    echo "[WARN] eval dataset not found: $GOLD_DATASET_PATH"
    echo "[SKIP] eval"
  fi
else
  echo "[SKIP] eval disabled"
fi

echo
echo "===================================================================================================="
echo "[OK] FULL SETUP COMPLETED"
echo "===================================================================================================="
echo "Run local API:"
echo "  ./run_local.sh"
echo
echo "Search test:"
echo "  ./run_search_test.sh"
echo "===================================================================================================="
