# LegalRAG

A browser-based RAG prototype for asking questions about local legal text
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
using cosine similarity. The three closest chunks are selected. If a selected
chunk is one part of a long block, all parts of that block are reconstructed
before the complete block is passed to Gemma through Ollama.

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
ollama pull gemma4:e4b-it-qat
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
HF_HOME=.cache/huggingface uvicorn backend.web:app
```

Open `http://127.0.0.1:8000` in a browser. The app loads
`documents/us_constitution.txt` and splits it at blank lines or runs of four
spaces. A block of 20 words or fewer is combined with the next block when the
result stays at 150 words or fewer. Longer blocks are split at sentence
boundaries into parts of about 150 words and reconstructed after retrieval.
The Constitution and amendments corpus was prepared from the official
[U.S. National Archives transcripts](https://www.archives.gov/founding-docs).

The 150-word limit is conservative for the 256-token input limit of
`all-MiniLM-L6-v2`.

## TODO

- Add a FastAPI backend so the RAG system can receive questions over HTTP
- Build an HTML, CSS, and JavaScript frontend for asking questions
- Show the retrieved source text with every answer
- Add clearer citations containing chunk, section, article, and amendment
  information
- Allow users to upload and index their own TXT documents
- Handle Ollama connection errors, loading delays, and timeouts clearly
- Improve Gemma 4 E4B QAT performance:
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
- Gemma 4 E4B QAT (`gemma4:e4b-it-qat`)
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
