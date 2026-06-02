#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

API_KEY="${DIFY_API_KEY:-change-me-ingest-very-long-key}"

curl -X POST "http://127.0.0.1:8000/search_base_articles" \
  -H "Content-Type: application/json" \
  -H "x-api-key: ${API_KEY}" \
  -d '{
    "query": "норматив стоимости одного квадратного метра общей площади жилого помещения",
    "top_k": 5,
    "search_mode": "weighted"
  }'
