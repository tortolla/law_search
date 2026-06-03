#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

export PYTHONPATH="${PYTHONPATH:-.}"

export GOLD_DATASET_PATH="${GOLD_DATASET_PATH:-data/processed/dataset_fixed.json}"
export EVAL_OUT_DIR="${EVAL_OUT_DIR:-reports/eval/weighted_doc_level}"

echo "===================================================================================================="
echo "RUN RETRIEVAL EVAL"
echo "===================================================================================================="
echo "GOLD_DATASET_PATH: $GOLD_DATASET_PATH"
echo "EVAL_OUT_DIR:      $EVAL_OUT_DIR"
echo "METHOD:            weighted"
echo "BM25_WEIGHT:       0.3"
echo "FRIDA_WEIGHT:      0.7"
echo "===================================================================================================="

if [[ ! -x ".venv/bin/python" ]]; then
  echo "[ERROR] .venv/bin/python not found"
  echo "Run setup first:"
  echo "  ./run_setup.sh"
  exit 1
fi

if [[ ! -f "$GOLD_DATASET_PATH" ]]; then
  echo "[ERROR] gold dataset not found: $GOLD_DATASET_PATH"
  exit 1
fi

mkdir -p "$EVAL_OUT_DIR"

exec .venv/bin/python scripts/08_eval_search.py \
  --dataset "$GOLD_DATASET_PATH" \
  --method weighted \
  --bm25-weight 0.3 \
  --frida-weight 0.7 \
  --candidate-k 1000 \
  --top-k 150 \
  --ks 1 5 10 15 20 25 35 50 70 90 120 150 \
  --out-dir "$EVAL_OUT_DIR"
