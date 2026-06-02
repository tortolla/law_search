#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

export PYTHONPATH="${PYTHONPATH:-.}"

DATASET="${GOLD_DATASET_PATH:-data/processed/dataset_fixed.json}"
OUT_DIR="reports/eval"
DETAILS_OUT="${OUT_DIR}/eval_weighted_details.csv"
SUMMARY_OUT="${OUT_DIR}/eval_weighted_summary.csv"
PLOT_OUT="${OUT_DIR}/doc_hit_curve.png"

mkdir -p "$OUT_DIR"

echo "===================================================================================================="
echo "RUN EVAL"
echo "===================================================================================================="
echo "DATASET:       $DATASET"
echo "DETAILS_OUT:   $DETAILS_OUT"
echo "SUMMARY_OUT:   $SUMMARY_OUT"
echo "PLOT_OUT:      $PLOT_OUT"
echo "METHOD:        weighted"
echo "BM25_WEIGHT:   0.3"
echo "FRIDA_WEIGHT:  0.7"
echo "===================================================================================================="

if [[ ! -x ".venv/bin/python" ]]; then
  echo "[ERROR] .venv/bin/python not found"
  echo "Run first:"
  echo "  ./run_setup.sh"
  exit 1
fi

if [[ ! -f "$DATASET" ]]; then
  echo "[ERROR] eval dataset not found: $DATASET"
  exit 1
fi

.venv/bin/python scripts/08_eval_search.py \
  --dataset "$DATASET" \
  --method weighted \
  --bm25-weight 0.3 \
  --frida-weight 0.7 \
  --candidate-k 1000 \
  --top-k 150 \
  --ks 1 5 10 15 20 25 35 50 70 90 120 150 \
  --details-out "$DETAILS_OUT" \
  --summary-out "$SUMMARY_OUT"

.venv/bin/python - <<'PY'
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

summary_path = Path("reports/eval/eval_weighted_summary.csv")
plot_path = Path("reports/eval/doc_hit_curve.png")

if not summary_path.exists():
    raise SystemExit(f"[ERROR] summary not found: {summary_path}")

df = pd.read_csv(summary_path)

if len(df) == 0:
    raise SystemExit("[ERROR] empty summary csv")

row = df.iloc[0]

ks = [1, 5, 10, 15, 20, 25, 35, 50, 70, 90, 120, 150]
values = []

for k in ks:
    col = f"doc_hit@{k}"
    if col not in row.index:
        raise SystemExit(f"[ERROR] missing column: {col}")
    values.append(float(row[col]))

plt.figure(figsize=(9, 5))
plt.plot(ks, values, marker="o")
plt.xlabel("Top-K")
plt.ylabel("Document Hit@K")
plt.title("Document-level retrieval quality")
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.05)
plt.xticks(ks, rotation=45)
plt.tight_layout()
plt.savefig(plot_path, dpi=200)

print("[OK] plot saved:", plot_path)
print()
print("DOC HIT CURVE:")
for k, v in zip(ks, values):
    print(f"top-{k:<3} {v:.6f}")
PY

echo
echo "===================================================================================================="
echo "[OK] EVAL COMPLETED"
echo "===================================================================================================="
echo "Summary: $SUMMARY_OUT"
echo "Details: $DETAILS_OUT"
echo "Plot:    $PLOT_OUT"
