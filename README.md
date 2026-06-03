[README.md](https://github.com/user-attachments/files/28545045/README.md)
# Law Search

Локальный гибридный поиск по корпусу нормативных документов.

Кратко, что делает проект:

- скачивает корпус с Яндекс.Диска;
- режет документы на смысловые чанки;
- строит BM25 lexical index;
- считает FRIDA dense embeddings;
- поднимает Milvus vector database;
- запускает FastAPI HTTP API;
- проверяет поиск через smoke-test;
- считает document-level retrieval eval.

---

# 1. Быстрый запуск с нуля

## 1.1. Системные зависимости на Ubuntu/Debian

```bash
apt update
apt install -y python3 python3-venv python3-pip curl git docker.io docker-compose screen
systemctl start docker
systemctl enable docker
```

Проверка:

```bash
python3 --version
docker --version
docker-compose --version
```

---

## 1.2. Скачать проект

```bash
git clone https://github.com/tortolla/law_search.git
cd law_search
cp .env.example .env
```

---

## 1.3. Запустить полный setup

Рекомендуется запускать через `screen`, потому что полный setup может идти долго.

```bash
screen -S law_setup
```

Внутри `screen`:

```bash
cd law_search
./run_setup.sh 2>&1 | tee setup.log
```

Выйти из `screen`, не останавливая процесс:

```text
Ctrl+A
D
```

Вернуться:

```bash
screen -r law_setup
```

Смотреть лог без входа в `screen`:

```bash
tail -f setup.log
```

Успешное завершение:

```text
[OK] FULL SETUP COMPLETED
```

---

# 2. Что делает `run_setup.sh`

`run_setup.sh` — главный скрипт полного развёртывания.

Он делает:

1. создаёт `.venv`;
2. ставит зависимости из `requirements.txt`;
3. скачивает или проверяет FRIDA model;
4. запускает Docker / Docker Compose;
5. поднимает Milvus stack: `etcd`, `minio`, `milvus-standalone`;
6. скачивает корпус в `data_big/`;
7. валидирует raw data;
8. строит `docs.parquet` и `chunks.parquet`;
9. строит BM25 index;
10. считает FRIDA embeddings;
11. загружает embeddings в Milvus collection;
12. валидирует индексы;
13. запускает retrieval eval.

Обычный запуск:

```bash
./run_setup.sh
```

Полезные режимы:

```bash
./run_setup.sh --skip-download
./run_setup.sh --skip-embeddings
./run_setup.sh --skip-eval
./run_setup.sh --force
```

Пример продолжения, если данные и embeddings уже есть, но нужно заново загрузить Milvus и посчитать eval:

```bash
./run_setup.sh --skip-download --skip-embeddings
```

---

# 3. Shell-скрипты

## `run_setup.sh`

Полный setup проекта с нуля.

```bash
./run_setup.sh
```

Использовать первым.

---

## `run_local.sh`

Запускает локальный FastAPI server.

```bash
./run_local.sh
```

По умолчанию сервер слушает:

```text
http://127.0.0.1:8000
```

Health-check:

```bash
curl "http://127.0.0.1:8000/health"
```

Ожидаемый ответ:

```json
{"ok":true,"service":"local_dify_bridge"}
```

---

## `run_search_test.sh`

Проверяет, что локальный поиск работает через API.

Перед запуском должен работать `run_local.sh`.

```bash
./run_search_test.sh
```

Ожидаемые признаки в JSON-ответе:

```json
"ok": true
"vector_backend": "milvus"
"method": "bm25_milvus_weighted"
"collection": "frida_chunks"
```

---

## `run_eval.sh`

Отдельно запускает retrieval evaluation по gold dataset.

```bash
./run_eval.sh
```

Результаты сохраняются в:

```text
reports/eval/weighted_doc_level/
```

---

# 4. Что генерируется после setup

## Raw data

```text
data_big/
```

Там лежат скачанные документы по категориям:

```text
construction_laws/
customs_laws/
energy_laws/
general_laws/
mining_laws/
oil_laws/
```

---

## Processed data

```text
data/processed/docs.parquet
data/processed/chunks.parquet
data/processed/docs.jsonl
data/processed/chunks.jsonl
data/processed/chunk_stats.json
```

---

## BM25 index

```text
data/indexes/bm25/bm25.pkl
data/indexes/bm25/bm25_info.json
```

---

## FRIDA embeddings

```text
data/indexes/frida/embeddings.npy
data/indexes/frida/model_info.json
```

---

## Milvus collection

По умолчанию используется collection:

```text
frida_chunks
```

Milvus endpoints:

```text
19530  Milvus gRPC
9091   Milvus health
9000   MinIO API
9001   MinIO console
```

Проверка контейнеров:

```bash
docker-compose ps
```

---

## Eval reports

```text
reports/eval/weighted_doc_level/
```

Основные файлы:

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

# 5. Валидационный датасет

В репозитории лежит gold dataset:

```text
data/processed/dataset_fixed.json
```

Он нужен для проверки качества поиска.

Важно:

- `dataset_fixed.json` лежит в GitHub;
- тяжёлые артефакты не лежат в GitHub;
- `chunks.parquet`, `embeddings.npy`, `bm25.pkl`, `data_big/`, `models/FRIDA/` генерируются локально через `run_setup.sh`.

---

# 6. Как проверить, что всё работает

## 6.1. Проверить setup

```bash
tail -n 100 setup.log
```

Должно быть:

```text
[OK] FULL SETUP COMPLETED
```

---

## 6.2. Проверить Milvus

```bash
docker-compose ps
```

Ожидаемо:

```text
etcd                healthy
minio               healthy
milvus-standalone   healthy
```

---

## 6.3. Проверить API

Терминал 1:

```bash
./run_local.sh
```

Терминал 2:

```bash
curl "http://127.0.0.1:8000/health"
```

---

## 6.4. Проверить поиск

```bash
./run_search_test.sh
```

Ожидаемо:

```json
"ok": true
"vector_backend": "milvus"
"method": "bm25_milvus_weighted"
```

---

## 6.5. Проверить eval

```bash
./run_eval.sh
```

Ожидаемо:

```text
[OK] retrieval evaluation completed
```

---

# 7. API для Dify / внешнего клиента

Основная ручка:

```text
POST /search_base_articles
```

Пример локального запроса:

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

Ответ содержит:

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

---

# 8. Запуск API для доступа извне

`run_local.sh` поднимает сервер на:

```text
127.0.0.1:8000
```

Это доступно только внутри сервера.

Если нужно открыть API наружу:

```bash
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

После этого внешний запрос:

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

Для production-доступа лучше использовать firewall, nginx/reverse proxy и длинный API key.

---

# 9. Как заменить корпус данных

Чтобы использовать другой корпус, нужно сохранить ожидаемую структуру `data_big/`.

## 9.1. Требуемая структура

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

Минимальные правила:

- верхние папки должны оканчиваться на `_laws`;
- внутри каждой категории должен быть `metadata.json`;
- текстовые документы должны лежать в папках, оканчивающихся на `_md`;
- документы должны быть в Markdown `.md`;
- `metadata.json` должен соответствовать документам категории.

Примеры валидных категорий:

```text
construction_laws
customs_laws
energy_laws
general_laws
mining_laws
oil_laws
```

---

## 9.2. Пересобрать всё под новый корпус

Удалить старые generated artifacts:

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

Положить новый корпус в `data_big/`.

Запустить полную пересборку:

```bash
./run_setup.sh --force
```

Если данные уже лежат локально и скачивать ничего не надо:

```bash
./run_setup.sh --skip-download --force
```

---

# 10. `.env`

Перед первым запуском:

```bash
cp .env.example .env
```

Основные параметры:

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

# 11. Что не лежит в GitHub

В GitHub не кладутся:

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

Они создаются локально через `run_setup.sh`.

---

# 12. Проверенная команда для руководителя

Минимальный сценарий проверки:

```bash
git clone https://github.com/tortolla/law_search.git
cd law_search
cp .env.example .env
./run_setup.sh
./run_local.sh
```

Во втором терминале:

```bash
cd law_search
./run_search_test.sh
```

Отдельно проверить eval:

```bash
./run_eval.sh
```

---

# 13. Как устроено разбиение на чанки

Разбиение документов не является тупой нарезкой по фиксированному числу символов.

Pipeline делает смысловое chunking-представление:

- документ разбивается на логические фрагменты;
- учитываются заголовки, страницы, абзацы и длинные куски;
- слишком длинные фрагменты дополнительно делятся с overlap;
- сохраняются `doc_id`, `chunk_id`, `chunk_ix`;
- в каждый chunk добавляется контекст документа;
- добавляются название документа, категория права, группа источника, раздел, ключевые слова;
- чанки становятся самодостаточными для retrieval;
- BM25 получает текст с метаданными;
- FRIDA embedding считается по тому же enriched chunk text.

То есть каждый chunk содержит не только фрагмент текста, но и навигационный/смысловой контекст:

```text
Категория права
Группа источника
Раздел источника
Ключевые слова
Документ
Текст фрагмента
```

Это повышает устойчивость поиска: запрос может попасть не только по точной фразе, но и по смыслу, названию документа, категории, ведомству или связанным ключевым словам.

---

# 14. Кратко по архитектуре поиска

Поиск работает так:

1. пользователь отправляет query;
2. BM25 ищет lexical candidates;
3. FRIDA кодирует query в dense vector;
4. Milvus ищет ближайшие vector candidates;
5. результаты объединяются weighted fusion;
6. API возвращает top-k chunks с текстом, scores и metadata.

Базовый режим:

```text
search_mode = weighted
bm25_weight = 0.3
frida_weight = 0.7
candidate_k = 1000
```

---

# 15. Основные признаки успешной установки

После `run_setup.sh`:

```text
[OK] FULL SETUP COMPLETED
```

После `run_search_test.sh`:

```json
"ok": true
"vector_backend": "milvus"
"method": "bm25_milvus_weighted"
"collection": "frida_chunks"
```

После `run_eval.sh`:

```text
[OK] retrieval evaluation completed
```
