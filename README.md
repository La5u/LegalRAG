# LegalRAG

A Linux-only Korean-first legal RAG prototype. It ingests Korean law HWPX, PDF,
and TXT files, stores chunks and embeddings in Neo4j, retrieves with vector
search, and generates cited answers through a local Ollama or vLLM model.

> Technical prototype only. Do not treat generated answers as legal advice.

## Pipeline

```text
HWPX, PDF, or TXT legal documents
 ↓
local text extraction
 ↓
chunking
 ↓
local embeddings
 ↓
Neo4j vector index
 ↓
local LLM answer with citations
```

## Requirements

- Linux
- Python 3.11 or newer
- Docker with Compose, or Podman with Compose
- Neo4j 5.26 Community
- Ollama

The included `docker-compose.yml` runs `neo4j:5.26-community`.

Neo4j uses two local ports:

- `7474`: Neo4j Browser at `http://127.0.0.1:7474`
- `7687`: Bolt driver connection used by the Python app

Default local Neo4j login:

```text
username: neo4j
password: legalrag-password
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` installs `torch` from the PyTorch CPU wheel index before
`sentence-transformers`, which avoids pulling NVIDIA CUDA packages on CPU-only
servers.

Download local models once:

```bash
HF_HOME=.cache/huggingface python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"
HF_HOME=.cache/huggingface python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
ollama pull qwen3.5:4b
ollama pull qwen2.5:1.5b-instruct
```

## Run

Start Neo4j, index changed documents, start Ollama if needed, and launch the
browser app:

```bash
./scripts/run
```

Open `http://127.0.0.1:8000`.

Lower-level commands:

```bash
./scripts/index
./scripts/dev
```

Use Podman for the local Neo4j container:

```bash
LEGALRAG_CONTAINER_ENGINE=podman ./scripts/index
LEGALRAG_CONTAINER_ENGINE=podman ./scripts/dev
LEGALRAG_CONTAINER_ENGINE=podman ./scripts/run
```

The Podman script path uses `podman run` directly for Neo4j, so it does not
depend on a working `podman compose` provider.

The default embedding model is `intfloat/multilingual-e5-small`. Use the same
embedding model for indexing and app runtime.

To use BGE-M3 instead:

```bash
LEGALRAG_EMBEDDING_MODEL="BAAI/bge-m3" ./scripts/run
```

`scripts/index` only starts Neo4j and indexes documents. `scripts/dev` starts
Neo4j, starts Ollama if needed, and launches the app without indexing. Use
`scripts/dev` when you know the index is already current and want faster
startup.

Run the local smoke checks after code changes:

```bash
LEGALRAG_EMBEDDING_MODEL="intfloat/multilingual-e5-small" python -m scripts.smoke
```

Common evaluation runs:

```bash
./scripts/eval curated retrieval
./scripts/eval codex retrieval
python -m evaluation.report --results /tmp/legalrag-eval/multilingual-e5-small_curated_retrieval.jsonl
```

## Docker Hosting Workflow

Docker or Podman is the deployment path, not the editing workflow. Keep working in this
repository locally, test changes, commit them, then rebuild and restart the
container on the server.

Typical local development:

```bash
./scripts/dev
```

Typical server update:

```bash
git pull
sudo docker compose build app
sudo docker compose up -d
```

Podman server update:

```bash
git pull
podman compose build app
podman compose up -d
```

The running website uses the code copied into the Docker image at build time.
Local edits do not appear on the website until the server rebuilds the image
and restarts the app container. Neo4j data stays in the Docker volume, so
replacing the app container does not delete the indexed database.

## How To Test

Quick backend smoke test:

```bash
LEGALRAG_EMBEDDING_MODEL="intfloat/multilingual-e5-small" python -m scripts.smoke
```

Faster variants:

```bash
python -m scripts.smoke --compile-only
python -m scripts.smoke --skip-write
python -m scripts.smoke --skip-lookups
```

Expected output:

```text
PASS backend modules compile
PASS Neo4j constraints exist
PASS temporary Document -> LegalUnit -> SourceChunk tree writes and cleans up
PASS 10 article graph lookups pass
```

Browser test:

```bash
./scripts/run
```

Open `http://127.0.0.1:8000`.

Try questions such as:

- `대한민국헌법 제10조에는 무엇이 있나요?`
- `개인정보의 수집ㆍ이용에 관한 조문은 몇 조인가요?`
- `자동화된 결정에 대한 권리는 어느 조문에 있나요?`

Manual startup:

```bash
docker compose up -d neo4j
ollama serve
source .venv/bin/activate
python -m uvicorn backend.web:app --host 127.0.0.1 --port 8000
```

With Podman, start Neo4j with:

```bash
podman volume exists legalrag_neo4j_data || podman volume create legalrag_neo4j_data
podman volume exists legalrag_neo4j_logs || podman volume create legalrag_neo4j_logs
podman run -d \
  --name legalrag-neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/${NEO4J_PASSWORD:-legalrag-password} \
  -v legalrag_neo4j_data:/data \
  -v legalrag_neo4j_logs:/logs \
  neo4j:5.26-community
```

vLLM startup:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B \
  --host 127.0.0.1 \
  --port 8001

LEGALRAG_LLM_PROVIDER=vllm \
LEGALRAG_GENERATION_MODEL=Qwen/Qwen3-32B \
LEGALRAG_VLLM_URL=http://127.0.0.1:8001 \
./scripts/dev
```

## Configuration

Default embedding model:

```bash
export LEGALRAG_EMBEDDING_MODEL="intfloat/multilingual-e5-small"
```

BGE-M3 option:

```bash
export LEGALRAG_EMBEDDING_MODEL="BAAI/bge-m3"
```

Embedding profiles set chunk and retrieval defaults automatically:

```text
BAAI/bge-m3:
  LEGALRAG_MAX_WORDS=260
  LEGALRAG_TOP_K=4
  LEGALRAG_EMBEDDING_BATCH_SIZE=16

intfloat/multilingual-e5-small:
  LEGALRAG_MAX_WORDS=140
  LEGALRAG_TOP_K=6
  LEGALRAG_EMBEDDING_BATCH_SIZE=64
```

Model tradeoff:

| Model | Use when | Tradeoff |
|---|---|---|
| `intfloat/multilingual-e5-small` | Fast local iteration and smoke testing | Lower semantic recall, but quick CPU indexing and retrieval |
| `BAAI/bge-m3` | Higher-quality multilingual legal retrieval experiments | Slower and heavier, but usually better semantic matching |
| `qwen3.5:4b` | Default local answer generation through Ollama | Practical for development, not a legal-quality benchmark |
| `qwen2.5:1.5b-instruct` | Faster lighter generation, used in evaluation scripts | Lower quality, faster |

Other useful settings:

```bash
export LEGALRAG_GENERATION_MODEL="qwen3.5:4b"
export LEGALRAG_LLM_PROVIDER="ollama"
export LEGALRAG_OLLAMA_THINK="false"
export LEGALRAG_VLLM_URL="http://127.0.0.1:8001"
export LEGALRAG_NEO4J_URI="bolt://127.0.0.1:7687"
export LEGALRAG_NEO4J_USER="neo4j"
export LEGALRAG_NEO4J_PASSWORD="legalrag-password"
export LEGALRAG_NEO4J_DATABASE="neo4j"
```

Neo4j data is stored in the Docker volume `legalrag_neo4j_data`. Logs are stored
in `legalrag_neo4j_logs`.

## Backup And Restore

Stop the app before backing up Neo4j data.

Create a compressed backup of the Neo4j Docker volume:

```bash
docker compose stop neo4j
docker run --rm \
  -v legalrag_neo4j_data:/data \
  -v "$PWD":/backup \
  alpine tar czf /backup/legalrag_neo4j_data.tar.gz -C /data .
docker compose start neo4j
```

With Podman, use `podman compose` and `podman run` for the same backup and
restore commands.

Restore that backup into the same volume:

```bash
docker compose stop neo4j
docker run --rm \
  -v legalrag_neo4j_data:/data \
  -v "$PWD":/backup \
  alpine sh -c "rm -rf /data/* && tar xzf /backup/legalrag_neo4j_data.tar.gz -C /data"
docker compose start neo4j
```

## Documents

Put Korean law HWPX files, selectable-text PDFs, or plain UTF-8 TXT files under
`documents/raw/`. The indexer reads this folder recursively and ignores
`documents/excluded/`. The current MVP assumes one file per law or source
document. Separate TXT article-title exports are not needed when the HWPX export
already includes article titles and lists.

PDF extraction uses `pypdfium2` by default. The loader also extracts with
`pdfplumber` and chooses that result when the PDFium text looks weak or when
pdfplumber preserves more legal text, such as table-heavy text or Hanja.

Use `./scripts/index` when documents, embedding model, or chunking settings change.
Existing unchanged documents are skipped using Neo4j metadata: source path, file
timestamp, embedding model, chunking version, and word limit.

The current chunking version is `word-block-v14-boundary-triples`. If this value changes
in code (`CHUNKING_VERSION` in `backend/config.py`), the next `./scripts/index` run
rebuilds affected documents.

## Retrieval Strategies

The `/ask` endpoint accepts an optional strategy in the JSON body:

```json
{"question": "개인정보의 수집ㆍ이용에 관한 조문은 몇 조인가요?", "strategy": "graph"}
```

Available strategies:

- `vector`: return only the top vector-search chunks.
- `graph`: GraphRAG mode. Start from vector hits, add nearby chunk parts, add
  graph facts from extracted concepts and relationships, and expand through
  `LegalUnit` nodes only when the question explicitly asks for article or
  section-level context.
- `neuro_symbolic`: Neuro-Symbolic mode. Start with GraphRAG evidence, then add
  an explicit symbolic reasoning trace that lists matched concepts, checked
  definitions, legal relations, exceptions, conditions, supporting chunks, and
  graph paths.

The browser app labels these modes as `Vector`, `GraphRAG`, and
`Neuro-Symbolic`.

## Current Scope

- Korean HWPX ingestion
- Neo4j vector storage and cosine retrieval
- Deterministic Neo4j legal-structure graph:
  `Document -> LegalUnit -[:SUPPORTED_BY]-> SourceChunk`
- Deterministic legal definition graph:
  `Term -> Definition -> LegalUnit`, with `SUPPORTED_BY` source evidence
- Kiwi-assisted legal relationship graph:
  `Concept -[:HAS_DUTY|MAY|PROHIBITS|MUST_NOTIFY|MAY_REQUEST]-> Concept`
- Legal connection graph:
  `Chunk -[:CITES]-> Citation -[:REFERS_TO]-> Document` and
  `Chunk -[:CITES]-> Citation -[:REFERS_TO_ARTICLE]-> LegalUnit`
- Exception edges between concepts: `Concept -[:EXCEPTION_TO]-> Concept`
- Neuro-symbolic trace categorizes retrieved evidence into definitions,
  obligations, prohibitions, permissions, exceptions, and conditions
  at query time without separate graph node labels
- Local embeddings through `sentence-transformers`
- Local answer generation through Ollama
- Swappable retrieval strategies: `vector`, GraphRAG-backed `graph`, and
  trace-backed `neuro_symbolic`

## Architecture Decisions

- **Standalone Neo4j GraphRAG** — Haystack was removed because only three
  helper types were used from it, not pipelines or document stores.
- **Deterministic graph construction** — Legal-unit, definition, case, and
  citation nodes are built from text structure patterns and regex, not from
  an LLM.
- **Neuro-symbolic traces** — The `neuro_symbolic` retrieval strategy adds a
  symbolic reasoning trace (definitions checked, relations found, conclusion
  type) as prompt context for the LLM.
- **Retrieval strategies** — `vector`, `graph` (GraphRAG), and
  `neuro_symbolic` are swappable via the `/ask` endpoint.
- **PDF extraction** — `pypdfium2` is the primary engine; `pdfplumber`
  serves as a fallback when pdfium output is short, has replacement
  characters, or misses table-heavy text and Hanja.
- **No browser upload** — Filesystem-based document management is simpler
  for sensitive on-premise legal use.
- **Evaluation** — The `evaluation/` module benchmarks retrieval precision
  and answer quality across strategies.
