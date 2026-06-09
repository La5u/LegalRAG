# LegalRAG

A terminal-based RAG prototype for asking questions about local legal text
documents.

> This is a technical prototype, not legal advice. Answers must be verified
> against the original documents.

## Pipeline

```text
TXT
 ↓
Text Cleaning
 ↓
Chunking
 ↓
Embeddings
 ↓
Document Store
```

At query time, the question is embedded and compared with the stored vectors
using cosine similarity. The three closest chunks are passed to Gemma through
Ollama.

## Setup

Requirements:

- Python 3.11 or newer
- [Ollama](https://ollama.com/)

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Download the embedding model into the project cache:

```bash
HF_HOME=.cache/huggingface python -c \
  "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

Download the Gemma model:

```bash
ollama pull gemma4:12b-it-q4_K_M
```

The model downloads are only required once.

## Run

Start Ollama:

```bash
ollama serve
```

In another terminal:

```bash
source .venv/bin/activate
python -m backend.main
```

The prototype loads `documents/us_constitution.txt`, creates 100-word chunks
with a 20-word overlap, embeds them, and accepts questions through `input()`.
The Constitution and amendments corpus was prepared from the official
[U.S. National Archives transcripts](https://www.archives.gov/founding-docs).

The original 500-word/100-word-overlap plan was reduced because
`all-MiniLM-L6-v2` truncates long inputs. With 500-word chunks, later text was
not represented in the vector and relevant passages were missed.

## TODO

- Add a FastAPI backend so the RAG system can receive questions over HTTP
- Build an HTML, CSS, and JavaScript frontend for asking questions
- Show the retrieved source text with every answer
- Add clearer citations containing chunk, section, article, and amendment
  information
- Preserve document headings and other metadata during chunking
- Allow users to upload and index their own TXT documents
- Handle Ollama connection errors, loading delays, and timeouts clearly
- Improve Gemma 4 12B performance without replacing the model:
  - Preload Gemma before the first question and keep it resident in memory
  - Reduce prompt size by sending only the most useful retrieved text
  - Tune Ollama context length, batch size, thread count, and memory settings
  - Reduce competing RAM usage and avoid swapping while Gemma is running
  - Evaluate supported GPU acceleration or hardware with more memory
  - Measure model loading, prompt processing, and answer-generation times
- Save embeddings so documents do not need to be indexed after every restart
- Add automated tests for chunking, retrieval, and answer generation
- Evaluate retrieval quality using a set of Constitution questions
- Compare different chunk sizes, overlap sizes, and embedding models
- Add hybrid semantic and keyword search for exact legal terms
- Support PDF parsing and OCR after the TXT workflow is stable
- Replace the in-memory document store with Qdrant or OpenSearch for
  production use
- Add authentication, document permissions, audit logs, monitoring, and
  backups for business use

## Tech Stack

### Backend

- Python
- deepset Haystack
- FastAPI
- Ollama
- Gemma 4 12B (`gemma4:12b-it-q4_K_M`)
- `all-MiniLM-L6-v2` embeddings
- Haystack `InMemoryDocumentStore`

### Frontend

- HTML
- CSS
- JavaScript

## Production Upgrades

For business use, the prototype should be upgraded with:

- Qdrant or OpenSearch instead of the in-memory document store
- Hybrid vector and BM25 keyword search
- A multilingual, legal-domain embedding model
- OCR, layout-aware parsing, and legal-aware chunking
- Source citations, reranking, and retrieval-quality evaluation
- Authentication, role-based document access, encryption, and audit logs
- Persistent file storage, backups, monitoring, background workers, and
  scalable model serving
