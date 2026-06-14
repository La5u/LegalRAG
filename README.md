# LegalRAG

A browser-based RAG prototype for asking questions about local legal text
documents.

> This is a technical prototype, not legal advice. Answers must be verified
> against the original documents.

## Pipeline

```text
TXT or text-based PDF
 ↓
Text Extraction and Cleaning
 ↓
Chunking
 ↓
Embeddings
 ↓
Document Store
```

At query time, the question is embedded and compared with the stored vectors
using cosine similarity. The three closest chunks are selected. If a selected
chunk is one part of a long block, its immediately previous and next chunks are
also included when they belong to that same block. A first or last chunk has
only one neighbor. Each included chunk remains separate and receives a
temporary citation number based on its order in the context.
PDF sources also show the page number containing the chunk.

## Setup

Requirements:

- Python 3.11 or newer
- [Ollama](https://ollama.com/)

Create and activate a virtual environment.

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the embedding model into the project cache.

Linux or macOS:

```bash
HF_HOME=.cache/huggingface python -c \
  "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

Windows PowerShell:

```powershell
$env:HF_HOME = ".cache/huggingface"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

Download the Gemma model:

```bash
ollama pull gemma4:e4b-it-qat
```

The model downloads are only required once.

Normal runtime is local: the browser connects to FastAPI on `127.0.0.1:8000`,
and FastAPI connects to Ollama on `127.0.0.1:11434`. Haystack telemetry is
disabled, and Hugging Face and Transformers run in offline mode.

## Run

Start Ollama:

```bash
ollama serve
```

In another terminal, activate the environment and start FastAPI.

Linux or macOS:

```bash
source .venv/bin/activate
python -m uvicorn backend.web:app
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.web:app
```

Open `http://127.0.0.1:8000` in a browser. The app loads every `.txt` and `.pdf`
file in `documents/`. TXT files are read directly. `pdfplumber` extracts text
and word positions from each PDF page. A larger-than-normal vertical gap
between lines becomes a paragraph break. Standalone numeric page lines are
removed. Extraction happens during startup or upload, so the PDF is not parsed
again for each question.
The original PDF remains in `documents/` instead of being converted into a
second TXT file. This preserves the authoritative source and makes future page
citations or improved parsing possible.

Extracted text is split at blank lines or runs of four spaces. A block of 50
words or fewer is combined with the next block when the result stays at 150
words or fewer. Longer blocks are split at sentence boundaries into parts of
about 150 words. After retrieval, the app adds at most one preceding and one
following part from the same original block.
The Constitution and amendments corpus was prepared from the official
[U.S. National Archives transcripts](https://www.archives.gov/founding-docs).

The 150-word limit is conservative for the 256-token input limit of
`all-MiniLM-L6-v2`.

The app saves the Haystack in-memory document store, including embeddings, in
`.cache/document_store.json`. On startup, the cache is reused when it is newer
than every source document. Otherwise, all documents are indexed again and the
cache is replaced. Changing the embedding model or chunking code requires
manually deleting the cache. Successful uploads also refresh it.

Use the `+` button beside the question field to upload another UTF-8 `.txt` or
text-based `.pdf` file. The app saves it in `documents/`, immediately chunks
and embeds it, and makes it searchable without a restart. Existing filenames
cannot be overwritten through the upload interface. Select `Files` to see
every indexed filename and its chunk count.

Image-only scanned PDFs have no text layer, so they require OCR. OCR means
software that recognizes text inside images and remains a future upgrade.

## Retrieval Limitation

The local neighbor window reduces the amount of text sent to Gemma, but it can
miss relevant context elsewhere in a long paragraph. This is especially
possible when a definition, condition, or exception appears several chunks
away from the retrieved match.

A production system should evaluate broader context strategies. Alternatives
include also providing the first and last chunks of the paragraph, dynamically
expanding the window until a section boundary, reranking all paragraph chunks,
or generating and retrieving a separate summary of each paragraph. These
approaches use more context or processing, so their retrieval quality and
latency should be measured on representative legal questions.

## Tech Stack

### Backend

- Python
- deepset Haystack
- FastAPI
- Ollama
- Gemma 4 E4B QAT (`gemma4:e4b-it-qat`)
- `all-MiniLM-L6-v2` embeddings
- Haystack `InMemoryDocumentStore`
- `pdfplumber` for local PDF text extraction

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
