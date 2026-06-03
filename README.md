## Production-ready local retrieval service for a legal document corpus.

The project provides:

- corpus download from Yandex Disk;
- document validation and preprocessing;# Law Search

Production-ready local retrieval service for a legal document corpus.

The project provides:

- corpus download from Yandex Disk;
- document validation and preprocessing;
- semantic chunking with metadata enrichment;
- BM25 lexical index;
- FRIDA dense embeddings;
- Milvus vector database;
- hybrid BM25 + dense retrieval;
- FastAPI HTTP API for local / Dify integration;
- document-level retrieval evaluation.

---

## 1. Quick start

### 1.1. Install system dependencies

Ubuntu/Debian:

```bash
apt update
apt install -y python3 python3-venv python3-pip curl git docker.io docker-compose
systemctl start docker
systemctl enable docker
```

Check:

```bash
python3 --version
docker --version
docker-compose --version
```

---

### 1.2. Clone repository

```bash
git clone https://github.com/tortolla/law_search.git
cd law_search
cp .env.example .env
```

Edit `.env` only if you need to change API keys, data URL, Milvus host/port, or collection name.

---

### 1.3. Run full setup

```bash
./run_setup.sh
```

Successful completion:

```text
[OK] FULL SETUP COMPLETED
```

The setup may take a long time because it downloads the corpus, builds chunks, computes FRIDA embeddings, loads vectors into Milvus, and runs evaluation.

---

## 2. Main scripts

### `run_setup.sh`

Full pipeline from repository checkout to working Milvus-backed search.

```bash
./run_setup.sh
```

Main stages:

1. creates `.venv`;
2. installs Python dependencies;
3. downloads/checks FRIDA model;
4. starts Docker Compose stack;
5. starts Milvus, MinIO, etcd;
6. downloads raw corpus into `data_big/`;
7. validates raw data;
8. builds `docs.parquet` and `chunks.parquet`;
9. builds BM25 index;
10. computes FRIDA embeddings;
11. loads vectors into Milvus;
12. validates indexes;
13. runs retrieval evaluation.

Useful flags:

```bash
./run_setup.sh --skip-download
./run_setup.sh --skip-embeddings
./run_setup.sh --skip-eval
./run_setup.sh --force
```

Typical continuation run when data, chunks, BM25, and embeddings already exist but Milvus should be rebuilt and eval should be rerun:

```bash
./run_setup.sh --skip-download --skip-embeddings
```

---

### `run_local.sh`

Starts the local FastAPI server.

```bash
./run_local.sh
```

Default address:

```text
http://127.0.0.1:8000
```

Health check:

```bash
curl "http://127.0.0.1:8000/health"
```

Expected response:

```json
{"ok":true,"service":"local_dify_bridge"}
```

---

### `run_search_test.sh`

Runs a smoke test against the local search API.

The API must already be running through `run_local.sh`.

```bash
./run_search_test.sh
```

Expected response fields:

```json
"ok": true
"vector_backend": "milvus"
"method": "bm25_milvus_weighted"
"collection": "frida_chunks"
```

---

### `run_eval.sh`

Runs document-level retrieval evaluation on the gold dataset.

```bash
./run_eval.sh
```

Output directory:

```text
reports/eval/weighted_doc_level/
```

---

## 3. Generated artifacts

The repository does not store heavy generated artifacts. They are created by `run_setup.sh`.

### Raw corpus

```text
data_big/
```

Expected top-level categories:

```text
construction_laws/
customs_laws/
energy_laws/
general_laws/
mining_laws/
oil_laws/
```

### Processed corpus

```text
data/processed/docs.parquet
data/processed/chunks.parquet
data/processed/docs.jsonl
data/processed/chunks.jsonl
data/processed/chunk_stats.json
```

### BM25 index

```text
data/indexes/bm25/bm25.pkl
data/indexes/bm25/bm25_info.json
```

### FRIDA embeddings

```text
data/indexes/frida/embeddings.npy
data/indexes/frida/model_info.json
```

### Milvus

Default collection:

```text
frida_chunks
```

Docker services:

```bash
docker-compose ps
```

Expected services:

```text
etcd
minio
milvus-standalone
```

Ports:

```text
19530  Milvus gRPC
9091   Milvus health
9000   MinIO API
9001   MinIO console
```

### Eval reports

```text
reports/eval/weighted_doc_level/
```

Main files:

```text
overall.csv
details.csv
by_query_type.csv
by_difficulty.csv
by_category.csv
doc_curve.csv
run_config.json
plots/doc_accuracy_curve.png
plots/doc_error_curve.png
```

---

## 4. Gold evaluation dataset

The repository includes:

```text
data/processed/dataset_fixed.json
```

It is used by `run_eval.sh` and by the evaluation stage inside `run_setup.sh`.

Important:

- `dataset_fixed.json` is committed to Git;
- `data_big/`, `chunks.parquet`, `docs.parquet`, `bm25.pkl`, `embeddings.npy`, and `models/FRIDA/` are generated locally;
- evaluation is primarily document-level; chunk-level metrics may be invalid after rechunking.

---

## 5. API usage

### 5.1. Search endpoint

```text
POST /search_base_articles
```

Example:

```bash
API_KEY=$(grep '^DIFY_API_KEY=' .env | cut -d '=' -f2-)

curl -s -X POST "http://127.0.0.1:8000/search_base_articles" \
  -H "Content-Type: application/json" \
  -H "x-api-key: ${API_KEY}" \
  -d '{
    "query": "норматив стоимости одного квадратного метра общей площади жилого помещения",
    "top_k": 5,
    "search_mode": "weighted"
  }'
```

Response contains:

```text
rank
doc_id
title
category
chunk_id
chunk_ix
hybrid_score
bm25_score
frida_score
chunk_text
method
dense_backend
collection
```

### 5.2. External access

`run_local.sh` binds to localhost:

```text
127.0.0.1:8000
```

For external clients or Dify deployment, start FastAPI on all interfaces:

```bash
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

External request:

```bash
curl -X POST "http://<SERVER_IP>:8000/search_base_articles" \
  -H "Content-Type: application/json" \
  -H "x-api-key: <DIFY_API_KEY>" \
  -d '{
    "query": "текст запроса",
    "top_k": 5,
    "search_mode": "weighted"
  }'
```

For production use, put the service behind firewall rules, nginx/reverse proxy, and a strong API key.

---

## 6. Verification checklist

After `run_setup.sh`:

```bash
docker-compose ps
```

Expected:

```text
etcd                healthy
minio               healthy
milvus-standalone   healthy
```

Check generated files:

```bash
ls -lh data/processed/docs.parquet
ls -lh data/processed/chunks.parquet
ls -lh data/indexes/bm25/bm25.pkl
ls -lh data/indexes/frida/embeddings.npy
```

Start API:

```bash
./run_local.sh
```

In another terminal:

```bash
curl "http://127.0.0.1:8000/health"
./run_search_test.sh
```

Run eval separately:

```bash
./run_eval.sh
```

Expected final markers:

```text
[OK] FULL SETUP COMPLETED
[OK] retrieval evaluation completed
```

Expected search response fields:

```json
"ok": true
"vector_backend": "milvus"
"method": "bm25_milvus_weighted"
"collection": "frida_chunks"
```

---

## 7. Replacing the corpus

To use another corpus, keep the same raw-data structure.

### 7.1. Required structure

```text
data_big/
  construction_laws/
    metadata.json
    <source_group>_md/
      <document_id>.md
      ...
  customs_laws/
    metadata.json
    <source_group>_md/
      <document_id>.md
      ...
```

Rules:

- top-level corpus directories must end with `_laws`;
- each category must contain `metadata.json`;
- Markdown documents must be located under folders ending with `_md`;
- documents must be `.md`;
- `metadata.json` must correspond to the documents in that category.

Valid category examples:

```text
construction_laws
customs_laws
energy_laws
general_laws
mining_laws
oil_laws
```

### 7.2. Full rebuild for a new local corpus

Remove generated artifacts:

```bash
rm -rf data_big/*
touch data_big/.gitkeep

rm -f data/processed/docs.parquet
rm -f data/processed/chunks.parquet
rm -f data/processed/docs.jsonl
rm -f data/processed/chunks.jsonl
rm -f data/processed/chunk_stats.json

rm -rf data/indexes/bm25/*
touch data/indexes/bm25/.gitkeep

rm -rf data/indexes/frida/*
touch data/indexes/frida/.gitkeep
```

Place the new corpus into `data_big/`.

Run:

```bash
./run_setup.sh --skip-download --force
```

If the corpus should be downloaded from a new Yandex Disk folder, set in `.env`:

```env
PUBLIC_DATA_URL=<new_public_yandex_disk_url>
```

Then run:

```bash
./run_setup.sh --force
```

---

## 8. Environment configuration

Create `.env` from the template:

```bash
cp .env.example .env
```

Core variables:

```env
DIFY_API_KEY=change-me-ingest-very-long-key
DIFY_INGEST_API_KEY=change-me-ingest-very-long-key
DIFY_RESULT_API_KEY=change-me-result-very-long-key

PUBLIC_DATA_URL=https://disk.yandex.ru/d/IcLlGxelh0A8GQ

RAW_DATA_DIR=data_big
PROCESSED_DIR=data/processed
INDEXES_DIR=data/indexes

FRIDA_MODEL_ID=ai-forever/FRIDA
FRIDA_MODEL_PATH=models/FRIDA

VECTOR_BACKEND=milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=frida_chunks

GOLD_DATASET_PATH=data/processed/dataset_fixed.json
```

---

## 9. Git policy

The repository should contain code, configs, scripts, README, and the gold dataset only.

Do not commit:

```text
.env
.venv/
data_big/
models/FRIDA/
data/processed/chunks.parquet
data/processed/docs.parquet
data/indexes/bm25/bm25.pkl
data/indexes/frida/embeddings.npy
reports/
results/
*.log
```

These files are generated by setup.

---

## 10. Chunking logic

The chunking stage is designed for retrieval, not for raw storage.

The pipeline:

- parses Markdown documents into logical text units;
- preserves document-level metadata;
- uses headings, pages, paragraphs, and long-piece splitting;
- applies overlap for long fragments;
- assigns stable `doc_id`, `chunk_id`, and `chunk_ix`;
- enriches each chunk with document title, legal category, source group, source section, and keywords;
- writes enriched text into `chunks.parquet` and `chunks.jsonl`;
- feeds the same enriched chunk text into BM25 and FRIDA.

A chunk contains both text and retrieval context:

```text
Legal category
Source group
Source section
Keywords
Document title
Chunk text
```

This improves retrieval robustness: a query can match by exact phrase, semantic content, document title, authority/source, legal category, or keywords.

---

## 11. Retrieval architecture

Default mode:

```text
search_mode = weighted
bm25_weight = 0.3
frida_weight = 0.7
candidate_k = 1000
```

Flow:

1. request enters FastAPI;
2. BM25 retrieves lexical candidates;
3. FRIDA encodes the query;
4. Milvus retrieves dense vector candidates;
5. scores are normalized and combined;
6. API returns ranked chunks with scores and metadata.

---

## 12. Minimal operational sequence

Full deployment:

```bash
git clone https://github.com/tortolla/law_search.git
cd law_search
cp .env.example .env
./run_setup.sh
```

Run API:

```bash
./run_local.sh
```

Search smoke test:

```bash
./run_search_test.sh
```

Evaluation:

```bash
./run_eval.sh
```

- semantic chunking with metadata enrichment;
- BM25 lexical index;
- FRIDA dense embeddings;
- Milvus vector database;
- hybrid BM25 + dense retrieval;
- FastAPI HTTP API for local / Dify integration;
- document-level retrieval evaluation.

---

## 1. Quick start

### 1.1. Install system dependencies

Ubuntu/Debian:

```bash
apt update
apt install -y python3 python3-venv python3-pip curl git docker.io docker-compose
systemctl start docker
systemctl enable docker
```

Check:

```bash
python3 --version
docker --version
docker-compose --version
```

---

### 1.2. Clone repository

```bash
git clone https://github.com/tortolla/law_search.git
cd law_search
cp .env.example .env
```

Edit `.env` only if you need to change API keys, data URL, Milvus host/port, or collection name.

---

### 1.3. Run full setup

```bash
./run_setup.sh
```

Successful completion:

```text
[OK] FULL SETUP COMPLETED
```

The setup may take a long time because it downloads the corpus, builds chunks, computes FRIDA embeddings, loads vectors into Milvus, and runs evaluation.

---

## 2. Main scripts

### `run_setup.sh`

Full pipeline from repository checkout to working Milvus-backed search.

```bash
./run_setup.sh
```

Main stages:

1. creates `.venv`;
2. installs Python dependencies;
3. downloads/checks FRIDA model;
4. starts Docker Compose stack;
5. starts Milvus, MinIO, etcd;
6. downloads raw corpus into `data_big/`;
7. validates raw data;
8. builds `docs.parquet` and `chunks.parquet`;
9. builds BM25 index;
10. computes FRIDA embeddings;
11. loads vectors into Milvus;
12. validates indexes;
13. runs retrieval evaluation.

Useful flags:

```bash
./run_setup.sh --skip-download
./run_setup.sh --skip-embeddings
./run_setup.sh --skip-eval
./run_setup.sh --force
```

Typical continuation run when data, chunks, BM25, and embeddings already exist but Milvus should be rebuilt and eval should be rerun:

```bash
./run_setup.sh --skip-download --skip-embeddings
```

---

### `run_local.sh`

Starts the local FastAPI server.

```bash
./run_local.sh
```

Default address:

```text
http://127.0.0.1:8000
```

Health check:

```bash
curl "http://127.0.0.1:8000/health"
```

Expected response:

```json
{"ok":true,"service":"local_dify_bridge"}
```

---

### `run_search_test.sh`

Runs a smoke test against the local search API.

The API must already be running through `run_local.sh`.

```bash
./run_search_test.sh
```

Expected response fields:

```json
"ok": true
"vector_backend": "milvus"
"method": "bm25_milvus_weighted"
"collection": "frida_chunks"
```

---

### `run_eval.sh`

Runs document-level retrieval evaluation on the gold dataset.

```bash
./run_eval.sh
```

Output directory:

```text
reports/eval/weighted_doc_level/
```

---

## 3. Generated artifacts

The repository does not store heavy generated artifacts. They are created by `run_setup.sh`.

### Raw corpus

```text
data_big/
```

Expected top-level categories:

```text
construction_laws/
customs_laws/
energy_laws/
general_laws/
mining_laws/
oil_laws/
```

### Processed corpus

```text
data/processed/docs.parquet
data/processed/chunks.parquet
data/processed/docs.jsonl
data/processed/chunks.jsonl
data/processed/chunk_stats.json
```

### BM25 index

```text
data/indexes/bm25/bm25.pkl
data/indexes/bm25/bm25_info.json
```

### FRIDA embeddings

```text
data/indexes/frida/embeddings.npy
data/indexes/frida/model_info.json
```

### Milvus

Default collection:

```text
frida_chunks
```

Docker services:

```bash
docker-compose ps
```

Expected services:

```text
etcd
minio
milvus-standalone
```

Ports:

```text
19530  Milvus gRPC
9091   Milvus health
9000   MinIO API
9001   MinIO console
```

### Eval reports

```text
reports/eval/weighted_doc_level/
```

Main files:

```text
overall.csv
details.csv
by_query_type.csv
by_difficulty.csv
by_category.csv
doc_curve.csv
run_config.json
plots/doc_accuracy_curve.png
plots/doc_error_curve.png
```

---

## 4. Gold evaluation dataset

The repository includes:

```text
data/processed/dataset_fixed.json
```

It is used by `run_eval.sh` and by the evaluation stage inside `run_setup.sh`.

Important:

- `dataset_fixed.json` is committed to Git;
- `data_big/`, `chunks.parquet`, `docs.parquet`, `bm25.pkl`, `embeddings.npy`, and `models/FRIDA/` are generated locally;
- evaluation is primarily document-level; chunk-level metrics may be invalid after rechunking.

---

## 5. API usage

### 5.1. Search endpoint

```text
POST /search_base_articles
```

Example:

```bash
API_KEY=$(grep '^DIFY_API_KEY=' .env | cut -d '=' -f2-)

curl -s -X POST "http://127.0.0.1:8000/search_base_articles" \
  -H "Content-Type: application/json" \
  -H "x-api-key: ${API_KEY}" \
  -d '{
    "query": "норматив стоимости одного квадратного метра общей площади жилого помещения",
    "top_k": 5,
    "search_mode": "weighted"
  }'
```

Response contains:

```text
rank
doc_id
title
category
chunk_id
chunk_ix
hybrid_score
bm25_score
frida_score
chunk_text
method
dense_backend
collection
```

### 5.2. External access

`run_local.sh` binds to localhost:

```text
127.0.0.1:8000
```

For external clients or Dify deployment, start FastAPI on all interfaces:

```bash
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

External request:

```bash
curl -X POST "http://<SERVER_IP>:8000/search_base_articles" \
  -H "Content-Type: application/json" \
  -H "x-api-key: <DIFY_API_KEY>" \
  -d '{
    "query": "текст запроса",
    "top_k": 5,
    "search_mode": "weighted"
  }'
```

For production use, put the service behind firewall rules, nginx/reverse proxy, and a strong API key.

---

## 6. Verification checklist

After `run_setup.sh`:

```bash
docker-compose ps
```

Expected:

```text
etcd                healthy
minio               healthy
milvus-standalone   healthy
```

Check generated files:

```bash
ls -lh data/processed/docs.parquet
ls -lh data/processed/chunks.parquet
ls -lh data/indexes/bm25/bm25.pkl
ls -lh data/indexes/frida/embeddings.npy
```

Start API:

```bash
./run_local.sh
```

In another terminal:

```bash
curl "http://127.0.0.1:8000/health"
./run_search_test.sh
```

Run eval separately:

```bash
./run_eval.sh
```

Expected final markers:

```text
[OK] FULL SETUP COMPLETED
[OK] retrieval evaluation completed
```

Expected search response fields:

```json
"ok": true
"vector_backend": "milvus"
"method": "bm25_milvus_weighted"
"collection": "frida_chunks"
```

---

## 7. Replacing the corpus

To use another corpus, keep the same raw-data structure.

### 7.1. Required structure

```text
data_big/
  construction_laws/
    metadata.json
    <source_group>_md/
      <document_id>.md
      ...
  customs_laws/
    metadata.json
    <source_group>_md/
      <document_id>.md
      ...
```

Rules:

- top-level corpus directories must end with `_laws`;
- each category must contain `metadata.json`;
- Markdown documents must be located under folders ending with `_md`;
- documents must be `.md`;
- `metadata.json` must correspond to the documents in that category.

Valid category examples:

```text
construction_laws
customs_laws
energy_laws
general_laws
mining_laws
oil_laws
```

### 7.2. Full rebuild for a new local corpus

Remove generated artifacts:

```bash
rm -rf data_big/*
touch data_big/.gitkeep

rm -f data/processed/docs.parquet
rm -f data/processed/chunks.parquet
rm -f data/processed/docs.jsonl
rm -f data/processed/chunks.jsonl
rm -f data/processed/chunk_stats.json

rm -rf data/indexes/bm25/*
touch data/indexes/bm25/.gitkeep

rm -rf data/indexes/frida/*
touch data/indexes/frida/.gitkeep
```

Place the new corpus into `data_big/`.

Run:

```bash
./run_setup.sh --skip-download --force
```

If the corpus should be downloaded from a new Yandex Disk folder, set in `.env`:

```env
PUBLIC_DATA_URL=<new_public_yandex_disk_url>
```

Then run:

```bash
./run_setup.sh --force
```

---

## 8. Environment configuration

Create `.env` from the template:

```bash
cp .env.example .env
```

Core variables:

```env
DIFY_API_KEY=change-me-ingest-very-long-key
DIFY_INGEST_API_KEY=change-me-ingest-very-long-key
DIFY_RESULT_API_KEY=change-me-result-very-long-key

PUBLIC_DATA_URL=https://disk.yandex.ru/d/IcLlGxelh0A8GQ

RAW_DATA_DIR=data_big
PROCESSED_DIR=data/processed
INDEXES_DIR=data/indexes

FRIDA_MODEL_ID=ai-forever/FRIDA
FRIDA_MODEL_PATH=models/FRIDA

VECTOR_BACKEND=milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=frida_chunks

GOLD_DATASET_PATH=data/processed/dataset_fixed.json
```

---

## 9. Git policy

The repository should contain code, configs, scripts, README, and the gold dataset only.

Do not commit:

```text
.env
.venv/
data_big/
models/FRIDA/
data/processed/chunks.parquet
data/processed/docs.parquet
data/indexes/bm25/bm25.pkl
data/indexes/frida/embeddings.npy
reports/
results/
*.log
```

These files are generated by setup.

---

## 10. Chunking logic

The chunking stage is designed for retrieval, not for raw storage.

The pipeline:

- parses Markdown documents into logical text units;
- preserves document-level metadata;
- uses headings, pages, paragraphs, and long-piece splitting;
- applies overlap for long fragments;
- assigns stable `doc_id`, `chunk_id`, and `chunk_ix`;
- enriches each chunk with document title, legal category, source group, source section, and keywords;
- writes enriched text into `chunks.parquet` and `chunks.jsonl`;
- feeds the same enriched chunk text into BM25 and FRIDA.

A chunk contains both text and retrieval context:

```text
Legal category
Source group
Source section
Keywords
Document title
Chunk text
```

This improves retrieval robustness: a query can match by exact phrase, semantic content, document title, authority/source, legal category, or keywords.

---

## 11. Retrieval architecture

Default mode:

```text
search_mode = weighted
bm25_weight = 0.3
frida_weight = 0.7
candidate_k = 1000
```

Flow:

1. request enters FastAPI;
2. BM25 retrieves lexical candidates;
3. FRIDA encodes the query;
4. Milvus retrieves dense vector candidates;
5. scores are normalized and combined;
6. API returns ranked chunks with scores and metadata.

---

## 12. Minimal operational sequence

Full deployment:

```bash
git clone https://github.com/tortolla/law_search.git
cd law_search
cp .env.example .env
./run_setup.sh
```

Run API:

```bash
./run_local.sh
```

Search smoke test:

```bash
./run_search_test.sh
```

Evaluation:

```bash
./run_eval.sh
```
