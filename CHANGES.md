# Changes

## Architecture

- Chose the assignment's Option B direction: standalone Neo4j vector RAG plus
  Neo4j GraphRAG.
- Changed the default local generation model to `qwen3.5:4b` for better Korean
  testing while keeping `LEGALRAG_GENERATION_MODEL` configurable.
- Disabled Ollama thinking mode by default with `LEGALRAG_OLLAMA_THINK=false`
  so Qwen streams visible answer tokens to the browser.
- Changed the default embedding model to `intfloat/multilingual-e5-small`;
  BGE-M3 remains available through `LEGALRAG_EMBEDDING_MODEL=BAAI/bge-m3`.
- Removed Haystack from the runtime code because the app was only using helper
  classes, not Haystack pipelines, components, or document stores.
- Replaced Haystack `Document` with a small local dataclass.
- Replaced `OllamaChatGenerator` with direct streaming calls to Ollama's
  `/api/chat` endpoint through `httpx`.
- Moved runtime configuration into `backend/config.py`.
- Added model-aware embedding profiles for BGE-M3 and multilingual E5-small.

## Storage And Retrieval

- Replaced the old in-memory document store and JSON cache with Neo4j.
- Added Neo4j constraints for `Document.source_path` and `Chunk.id`.
- Added a Neo4j constraint for `LegalUnit.id`.
- Added a Neo4j constraint for `Concept.name`.
- Added a Neo4j vector index for `Chunk.embedding`.
- Added deterministic legal-structure graph storage:
  `Document -CONTAINS-> LegalUnit -SUPPORTED_BY-> Chunk`.
- Added generic `LegalUnit.kind` values for chapter, section, article, clause,
  item, and sub-item structure.
- Added direct concept relationship storage for legal triples and legal
  connections.
- Added vector-index dimension checking so switching embedding models recreates
  the vector index when needed.
- Batched paragraph-neighbor lookup into one Cypher query instead of one query
  per retrieved chunk.
- Removed unused relationship-property index creation.
- Scoped document deletion and pruning so cleanup no longer starts by scanning
  every relationship in the Neo4j database.
- Moved repeated chunk record conversion and Cypher return fields out of
  `backend/rag.py`.

## Indexing And Speed

- Added `backend/index.py` for explicit indexing.
- Added `scripts/index` for explicit Linux indexing.
- Removed startup indexing from FastAPI.
- Lazy-loaded the embedding model so app startup does not load models.
- Kept unchanged-document skipping based on source path, file timestamp,
  embedding model, chunking version, and word limit.
- Added `evaluation.evaluator` for incremental JSONL evaluation across Vector,
  GraphRAG, and Neuro-Symbolic modes.
- Added source precision measurement with `context_precision` for each
  evaluation question.

## Ingestion

- Made ingestion HWPX-only for the current Korean MVP.
- Removed PDF and TXT ingestion from the active code path.
- Added local HWPX parsing through Python zip/XML parsing.
- Added Korean-aware sentence splitting for long chunks.
- Added Korean legal-unit marker detection for headings and list units like
  `제1조(목적)`, `①`, `1.`, and `가.`.
- Added legal-unit metadata to each chunk so indexing can build the structure
  graph without LLM extraction.
- Removed leftover page-number plumbing from the old PDF path.

## Frontend

- Removed browser upload UI.
- Kept question asking, source display, citation links, and indexed-file list.
- Changed the visible retrieval choices to `GraphRAG` and `Vector`.
- Added handling for malformed streamed JSON lines.

## Dependencies

- Removed `haystack-ai`.
- Removed `ollama-haystack`.
- Kept `pdfplumber` for table-heavy or weak PDF extraction fallback.
- Removed `python-multipart`.
- Added explicit `httpx`.
- Added `kiwipiepy` for Korean morphological analysis during triple extraction.
- Kept `fastapi`, `uvicorn`, `sentence-transformers`, and `neo4j`.

## Documents

- Removed old English/sample corpus files.
- Removed old screenshot image.
- Kept Korean HWPX legal documents as the active MVP corpus.

## Developer Scripts

- Added `scripts/dev` for Linux local startup.
- Added `scripts/run` as the one-command path: index changed documents, start
  Ollama if needed, and launch the app.
- Added customizable `backend.smoke` checks with `--compile-only`,
  `--skip-write`, and `--skip-lookups`.
- Changed Neo4j readiness checks to use Bolt connectivity instead of the Neo4j
  browser HTTP port.
- Made startup scripts exit when Ollama is missing instead of starting a
  half-working app.

## Documentation

- Rewrote `README.md` to focus on the current lean Neo4j/HWPX setup.
- Added cleanup justifications to `README.md`.
- Trimmed `TODO.md` to unfinished GraphRAG milestones, evaluation needs, and
  future refactor candidates.
- Updated docs to use `LegalUnit`, `Concept`, `GraphRAG`, and `Vector`
  terminology.
- Updated `AGENTS.md` to prefer a lean codebase.

## Validation Performed

- Ran Python compile checks for backend modules.
- Ran shell syntax checks for Bash scripts.
- Ran `pip check` after dependency removal.
- Smoke-tested FastAPI startup without automatic indexing.
- Smoke-tested `/documents` against local Neo4j.
- Verified `qwen3.5:4b` is installed in Ollama.
- Verified Neo4j has the `Document`, `Chunk`, `LegalUnit`, and `Concept` uniqueness
  constraints.
- Smoke-tested temporary `Document -> LegalUnit -> Chunk` tree writes and cleanup
  without loading BGE-M3 or reindexing the real corpus.
- Reindexed the Korean HWPX documents with
  `intfloat/multilingual-e5-small` for development-speed testing.
- Tested 10 article-focused graph questions:
  5 exact article lookups and 5 phrase-to-article lookups passed.
- Ran `LEGALRAG_EMBEDDING_MODEL=intfloat/multilingual-e5-small python -m backend.smoke`.
- Verified Qwen generation streams a visible first answer token with
  `think=false`.
- Verified GraphRAG retrieval adds graph facts for relationship questions such
  as `MAY_REQUEST`, `MUST_NOTIFY`, and `CITES`.
