# LegalRAG

A local Korean-first legal RAG prototype. It ingests Korean law HWPX, PDF, and
TXT files, stores chunks and embeddings in Neo4j, retrieves with vector search,
and generates cited answers through a local Ollama model.

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

- Python 3.11 or newer
- Docker with Compose
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

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks virtualenv activation, run this in the same terminal and
then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Download local models once:

```bash
HF_HOME=.cache/huggingface python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"
HF_HOME=.cache/huggingface python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
ollama pull qwen3.5:4b
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
LEGALRAG_EMBEDDING_MODEL="intfloat/multilingual-e5-small" python -m backend.smoke
```

## How To Test

Quick backend smoke test:

```bash
LEGALRAG_EMBEDDING_MODEL="intfloat/multilingual-e5-small" python -m backend.smoke
```

Faster variants:

```bash
python -m backend.smoke --compile-only
python -m backend.smoke --skip-write
python -m backend.smoke --skip-lookups
```

Expected output:

```text
PASS backend modules compile
PASS Neo4j constraints exist
PASS temporary Document -> LegalUnit -> Chunk tree writes and cleans up
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
| `qwen3.5:4b` | Local answer generation through Ollama | Practical for development, not a legal-quality benchmark |

Other useful settings:

```bash
export LEGALRAG_GENERATION_MODEL="qwen3.5:4b"
export LEGALRAG_OLLAMA_THINK="false"
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

The current chunking version is `word-block-v11`. If this value changes in code,
the next `./scripts/index` run rebuilds affected documents.

## Retrieval Strategies

The `/ask` endpoint accepts an optional strategy in the JSON body:

```json
{"question": "개인정보의 수집ㆍ이용에 관한 조문은 몇 조인가요?", "strategy": "hybrid"}
```

Available strategies:

- `vector`: return only the top vector-search chunks.
- `hybrid`: GraphRAG mode. Start from vector hits, add nearby chunk parts, add
  graph facts from extracted concepts and relationships, and expand through
  `LegalUnit` nodes only when the question explicitly asks for article or
  section-level context.

The browser app labels these modes as `Vector` and `GraphRAG`.

## Current Scope

- Korean HWPX ingestion
- Neo4j vector storage and cosine retrieval
- Deterministic Neo4j legal-structure graph:
  `Document -> LegalUnit -> Chunk`
- Deterministic legal definition graph:
  `Term -> Definition -> LegalUnit`, with chunk-level source evidence
- Kiwi-assisted legal relationship graph:
  `Concept -[:HAS_DUTY|MAY|PROHIBITS|MUST_NOTIFY|MAY_REQUEST]-> Concept`
- Deterministic legal connection graph:
  `Concept -[:CITES|EXCEPTION_TO]-> Concept`
- Local embeddings through `sentence-transformers`
- Local answer generation through Ollama
- Swappable retrieval strategies: `vector` and GraphRAG-backed `hybrid`

## Cleanup Decisions

This project uses the assignment's Option B: a standalone Neo4j GraphRAG
prototype. Haystack was removed because the current code only used three helper
types from it, not Haystack pipelines, components, or document stores. Removing
it keeps the code and dependencies smaller while preserving the required
baseline vector RAG query path.

PDF and browser upload support were also removed for now. The current MVP
corpus is HWPX and plain TXT legal text, and filesystem-based document
management is simpler for sensitive on-premise legal documents. PDF/OCR can be
added later after the HWPX/TXT pipeline and GraphRAG comparison are stable.

Next major step: evaluate `vector` and GraphRAG-backed `hybrid` retrieval with
context precision, answer quality, and relationship-reasoning questions. The
current GraphRAG path already adds Kiwi-assisted legal triples, citations, and
exception links as graph facts before generation.
