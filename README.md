# LegalRAG

A Linux-only Korean-first legal RAG prototype that runs entirely on your own
machine. It ingests Korean law HWPX, PDF, and TXT files, stores chunks and
embeddings in Neo4j, retrieves with vector search, and generates cited answers
through a local Ollama or vLLM model. After the models are downloaded, no
internet connection is needed and no document leaves the machine.

Originally developed by [Lasu](https://github.com/La5u). Licensed under
[Apache 2.0](LICENSE).

- **Quick start:** [docs/SETUP.md](docs/SETUP.md) (설치 안내)
- **Verify it offline yourself:** [docs/OFFLINE_TEST.md](docs/OFFLINE_TEST.md)
  (오프라인 검증)

> Technical prototype only. Do not treat generated answers as legal advice.

## Worked Example / 예시

The repository ships a fictional contract,
[`documents/raw/samples/샘플_소프트웨어개발용역계약서.txt`](documents/raw/samples/샘플_소프트웨어개발용역계약서.txt),
so you can try LegalRAG without downloading any legal documents. After
[setup](docs/SETUP.md), run `./scripts/run`. The other documents listed in the
manifest are skipped when they are not present. Then open
`http://127.0.0.1:8000`, choose `GraphRAG`, and ask:

> 샘플 소프트웨어 개발 용역 계약서에서 납품이 늦어지면 지체상금은 하루에 얼마이고
> 최대 한도는 얼마인가요?

The same question can be sent from a terminal:

```bash
curl -N http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "샘플 소프트웨어 개발 용역 계약서에서 납품이 늦어지면 지체상금은 하루에 얼마이고 최대 한도는 얼마인가요?", "strategy": "graph"}'
```

**Expected answer.** The late-delivery penalty is 2.5/1,000 of the contract
price per day of delay, and the total is capped at 8/100 of the contract price.
The wording will vary by model, but these two ratios should be in the answer.

> 지체상금은 지체일수 1일당 계약금액의 1천분의 2.5이며, 총액은 계약금액의 100분의 8을
> 초과할 수 없습니다 (제9조제1항·제2항).

**Supporting clause.** The answer should cite the sample file, and the cited
evidence should contain this text from **제9조(지체상금)**:

> ① 을이 제7조제1항의 납품기한까지 산출물을 납품하지 못한 경우, 을은 지체일수 1일당
> 계약금액의 1천분의 2.5에 해당하는 지체상금을 갑에게 지급하여야 한다.
>
> ② 제1항의 지체상금 총액은 계약금액의 100분의 8을 초과할 수 없다.

To check the citation, open the sample file and find 제9조. The contract price
in 제4조제1항 is ₩123,000,000, which works out to ₩307,500 per day and a cap
of ₩9,840,000. The values are deliberately unusual, so a correct answer has to
come from the contract, not from the model's general knowledge.

`./scripts/offline-check` asks this exact question with the network
disconnected and checks the answer and citation automatically. See
[docs/OFFLINE_TEST.md](docs/OFFLINE_TEST.md).

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

`requirements.txt` pins the tested direct dependencies and uses the PyTorch CPU
wheel index, which avoids pulling NVIDIA CUDA packages on CPU-only servers.

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

To use an OpenAI-compatible hosted generator, keep the API key in the
environment rather than a repository file:

```bash
export LEGALRAG_LLM_PROVIDER="openai_compatible"
export LEGALRAG_LLM_BASE_URL="https://api.example.com/v1"
export LEGALRAG_LLM_API_KEY="..."
export LEGALRAG_GENERATION_MODEL="your-model-name"
./scripts/dev
```

Hosted models are off by default, so a fresh install never sends data off the
machine. Setting `LEGALRAG_ALLOW_HOSTED_MODELS=true` adds an answer-model
dropdown to the web interface. `Local` uses the server-configured Ollama, vLLM,
or hosted model. The other choices use the API key entered by the browser user:

- `gemini-3.1-flash-lite` through the Gemini API
- `qwen/qwen3.6-27b` through the Groq API

These are fixed, free-tier-eligible choices; their quotas and availability are
controlled by the providers. The key is kept in browser `sessionStorage`, sent
only with hosted-model requests, and discarded when the browser tab closes.
Use HTTPS: each hosted request sends the question and retrieved evidence to the
selected provider.

`scripts/index` only starts Neo4j and indexes documents. `scripts/dev` starts
Neo4j, starts Ollama if needed, and launches the app without indexing. Use
`scripts/dev` when you know the index is already current and want faster
startup.

Run the local smoke checks after code changes:

```bash
LEGALRAG_INTEGRATION=1 python -m pytest -q tests/test_integration_smoke.py
```

Common evaluation runs:

```bash
./scripts/eval retrieval
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

Unit tests:

```bash
python -m pytest -q
```

Neo4j integration checks:

```bash
LEGALRAG_INTEGRATION=1 python -m pytest -q tests/test_integration_smoke.py
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
  -p 127.0.0.1:7474:7474 \
  -p 127.0.0.1:7687:7687 \
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

## Remote Slurm-Managed vLLM Inference

Instead of running the generation model locally, `scripts/remote-vllm` submits
an on-demand vLLM inference server as a Slurm job on a shared GPU server,
waits for it to start, and opens an SSH tunnel so the app can reach it at
`http://127.0.0.1:8001` like any other vLLM endpoint.

This exists because the GPU host is shared: only one Slurm GPU job can run
at a time, so the server must be requested through `sbatch`/`squeue`, not
launched directly, and it must give back the GPU when nobody is using it.

Remote-side files live in `deploy/remote/` and are copied to
`~/legalrag-infer/` on the GPU box:

- `infer_server.sbatch` — Slurm batch script that starts vLLM bound to
  `127.0.0.1` on the compute node, then watches its `/metrics` request
  counter and self-cancels the job (`scancel`) after
  `IDLE_TIMEOUT_SECONDS` (default 1800s) of no new requests, so the GPU
  doesn't sit idle. It also carries a `--time=04:00:00` cap as a hard
  backstop.
- `submit_if_needed.sh` — idempotent `sbatch` wrapper; reuses an existing
  queued/running `legalrag_infer` job instead of submitting a duplicate.

Local usage:

```bash
./scripts/remote-vllm up      # submit if needed, wait for RUNNING, open tunnel, wait for vLLM
./scripts/remote-vllm status  # squeue for the legalrag_infer job
./scripts/remote-vllm stop-job     # scancel the Slurm job
./scripts/remote-vllm stop-tunnel  # close the local SSH tunnel
```

To have `./scripts/run` / `./scripts/dev` do this automatically:

```bash
LEGALRAG_LLM_PROVIDER=vllm \
LEGALRAG_VLLM_REMOTE=1 \
LEGALRAG_GENERATION_MODEL=legalrag \
./scripts/dev
```

Since Slurm queues the job automatically when the GPU is busy with someone
else's work, `up` can be run any time — it just waits longer.

Required env vars: `LEGALRAG_VLLM_REMOTE_HOST`, `LEGALRAG_VLLM_REMOTE_USER`.
Optional env vars:
`LEGALRAG_VLLM_REMOTE_DIR` (`legalrag-infer`),
`LEGALRAG_VLLM_REMOTE_PORT` (`18080`),
`LEGALRAG_VLLM_LOCAL_PORT` (`8001`).

### In-process autostart (when the app runs on the Slurm node itself)

When the backend is deployed directly on the GPU/Slurm node there's no separate desktop to run `scripts/remote-vllm` from.
Setting `LEGALRAG_SLURM_AUTOSTART_VLLM=1` makes `backend/slurm_vllm.py` do
the equivalent check in-process, on the first request that needs local
generation:

- If vLLM is already reachable at `LEGALRAG_VLLM_URL`, use it immediately.
- Otherwise, check `squeue`. If another job (not ours) is queued or running,
  raise immediately — the request fails fast with a `gpu_busy` error rather
  than waiting behind someone else's job — so the app never competes
  with someone else's job for a shared GPU.
- If the queue is otherwise empty, submit (`submit_if_needed.sh`) and poll
  until vLLM answers, up to `LEGALRAG_SLURM_STARTUP_TIMEOUT` seconds
  (default 360s) — cold start takes a few minutes for a 32B model.

Env vars: `LEGALRAG_SLURM_AUTOSTART_VLLM` (`false`), `LEGALRAG_SLURM_JOB_NAME`
(`legalrag_infer`), `LEGALRAG_SLURM_SUBMIT_DIR` (`~/legalrag-infer`),
`LEGALRAG_SLURM_STARTUP_TIMEOUT` (`360`). Requires
`LEGALRAG_LLM_PROVIDER=vllm`.

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

The repository ships one fictional sample contract under
`documents/raw/samples/`. The rest of the corpus listed in
`documents/source_manifest.csv` is not redistributed, because its sources have
different terms. Registered files that are not present are skipped at indexing.
`scripts/download_legal_documents.py` fetches the public sources in
`documents/download_plan.json`; check each source's terms yourself.

Register Korean law HWPX files, selectable-text PDFs, or plain UTF-8 TXT files in
`documents/source_manifest.csv`, and place the assets under `documents/raw/`.
The manifest controls inclusion and supplies source metadata. HWPX and PDF
assets for the same legal version are treated as representations of one logical
document; HWPX supplies searchable text and the matching PDF supplies the
browser source when both exist.

PDF extraction uses `pypdfium2` by default. The loader also extracts with
`pdfplumber` and chooses that result when the PDFium text looks weak or when
pdfplumber preserves more legal text, such as table-heavy text or Hanja.

Use `./scripts/index` when documents, embedding model, or chunking settings change.
Existing unchanged documents are skipped using Neo4j metadata: source path, file
timestamp, embedding model, automatic indexing-pipeline fingerprint, and word
limit. The fingerprint is calculated from the parser, chunker, metadata, and
graph-indexing code, so relevant code changes trigger rebuilding without a
manual version edit.

## Retrieval Strategies

The `/ask` endpoint accepts an optional strategy in the JSON body:

```json
{"question": "개인정보의 수집ㆍ이용에 관한 조문은 몇 조인가요?", "strategy": "graph"}
```

Available strategies:

- `vector`: return only the top vector-search chunks.
- `graph`: GraphRAG mode. Combine vector and legal-text candidates, add nearby
  chunk parts, expose an extracted graph relation as cited evidence, and expand through
  `LegalUnit` nodes only when the question explicitly asks for article or
  section-level context.
- `neuro_symbolic`: Neuro-Symbolic mode. Start with GraphRAG evidence, then use
  an explicit symbolic trace to prioritize supporting evidence and list matched concepts, checked
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
- **Deterministic graph construction** — Legal-unit, definition, and case
  nodes are built from text structure patterns and regex, not from an LLM.
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

## License

Everything in this repository, including the code, documentation, sample
contract, and evaluation questions, is licensed under the
[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for copyright and
contributor credits.

The repository does not include the legal documents listed in
`documents/source_manifest.csv`. If you download them, they keep their own
terms, which are recorded in the manifest. Dependencies, models, and services
you install also have their own terms, including Neo4j Community (GPLv3).
