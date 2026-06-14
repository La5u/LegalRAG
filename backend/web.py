import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from backend.ingest import chunk_text, load_document
from backend.rag import LegalRAG

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_PATH = PROJECT_ROOT / "documents"
CACHE_PATH = PROJECT_ROOT / ".cache"
DOCUMENT_STORE_PATH = CACHE_PATH / "document_store.json"
SUPPORTED_SUFFIXES = {".pdf", ".txt"}


def get_document_paths() -> list[Path]:
    return sorted(
        path
        for path in DOCUMENTS_PATH.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def cache_is_valid(document_paths: list[Path]) -> bool:
    if not DOCUMENT_STORE_PATH.exists():
        return False

    cache_time = DOCUMENT_STORE_PATH.stat().st_mtime_ns
    return all(cache_time >= path.stat().st_mtime_ns for path in document_paths)


def save_cache(rag_system: LegalRAG) -> None:
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    rag_system.save_document_store(str(DOCUMENT_STORE_PATH))


def build_rag() -> LegalRAG:
    rag_system = LegalRAG()
    document_paths = get_document_paths()

    if cache_is_valid(document_paths):
        rag_system.load_document_store(str(DOCUMENT_STORE_PATH))
        return rag_system

    for document_path in document_paths:
        rag_system.index(
            chunk_text(
                load_document(document_path),
                document_name=document_path.name,
            )
        )
    save_cache(rag_system)
    return rag_system


rag = build_rag()
app = FastAPI()


class Question(BaseModel):
    question: str


@app.get("/")
def home() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "frontend" / "index.html")


@app.get("/documents")
def list_documents() -> dict:
    return {"documents": rag.list_documents()}


@app.post("/ask")
async def ask(request: Question) -> StreamingResponse:
    async def stream():
        chunks = rag.retrieve(request.question)
        sources = [
            {
                "document": chunk.meta["document_name"],
                "citation": citation_number,
                "chunk": chunk.meta["chunk_number"],
                "page": chunk.meta.get("page_number"),
                "score": chunk.score,
                "text": chunk.content,
            }
            for citation_number, chunk in enumerate(chunks, start=1)
        ]
        yield json.dumps({"type": "sources", "sources": sources}) + "\n"

        tokens = asyncio.Queue()

        async def send_token(chunk) -> None:
            await tokens.put(chunk.content)

        async def generate() -> None:
            try:
                await rag.generate_async(
                    request.question,
                    chunks,
                    send_token,
                )
            except Exception as error:
                await tokens.put(error)
            finally:
                await tokens.put(None)

        generation = asyncio.create_task(generate())

        while (token := await tokens.get()) is not None:
            if isinstance(token, Exception):
                yield json.dumps({"type": "error"}) + "\n"
                await generation
                return
            yield json.dumps({"type": "token", "text": token}) + "\n"

        await generation
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/documents")
def upload_document(file: UploadFile = File(...)) -> dict:
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if (
        not filename
        or Path(filename).name != filename
        or "\\" in filename
        or suffix not in SUPPORTED_SUFFIXES
    ):
        raise HTTPException(
            status_code=400,
            detail="Choose a TXT or PDF file with a safe filename.",
        )

    document_path = DOCUMENTS_PATH / filename
    if document_path.exists():
        raise HTTPException(
            status_code=409,
            detail="A document with that filename already exists.",
        )

    document_path.write_bytes(file.file.read())
    try:
        chunks = chunk_text(
            load_document(document_path),
            document_name=filename,
        )
        rag.index(chunks)
    except Exception:
        document_path.unlink(missing_ok=True)
        raise
    save_cache(rag)

    return {
        "document": filename,
        "chunks": len(chunks),
    }
