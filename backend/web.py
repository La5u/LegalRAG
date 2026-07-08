import asyncio
import json
import re
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.document_types import document_type_from_source_path
from backend.rag import LegalRAG
from backend.retrieval import RetrievalUnavailable
from backend.routing import route_reason

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DOCUMENTS_PATH = PROJECT_ROOT / "documents" / "raw"

CATEGORIES = [
    "all",
    "statute",
    "regulation",
    "precedent",
    "constitutional_decision",
    "contract",
    "guidance",
    "qna",
]


def _normalized_document_stem(path: Path) -> str:
    stem = re.sub(r"\([^)]*\)", "", path.stem).strip()
    return stem.casefold()


def _safe_raw_document_path(source_path: str) -> Path | None:
    path = (PROJECT_ROOT / source_path).resolve()
    try:
        path.relative_to(RAW_DOCUMENTS_PATH)
    except ValueError:
        return None
    return path


def _matching_pdf_path(path: Path) -> Path | None:
    if path.suffix.lower() == ".pdf" and path.exists():
        return path

    normalized_stem = _normalized_document_stem(path)
    for candidate in path.parent.glob("*.pdf"):
        if _normalized_document_stem(candidate) == normalized_stem:
            return candidate
    return None


def document_pdf_url(source_path: str | None) -> str | None:
    if not source_path:
        return None

    path = _safe_raw_document_path(source_path)
    if path is None:
        return None

    pdf_path = _matching_pdf_path(path)
    if pdf_path is None:
        return None

    relative_path = pdf_path.relative_to(RAW_DOCUMENTS_PATH).as_posix()
    return f"/documents/raw/{quote(relative_path, safe='/')}"


def graph_fact_rows(chunks) -> list[dict]:
    rows = []
    for chunk in chunks:
        for fact in chunk.meta.get("graph_facts", []):
            rows.append(fact.to_response())
    return rows


def add_trace_citations(
    trace_payload: dict | None,
    sources: list[dict],
    category: str,
) -> dict | None:
    if not trace_payload:
        return None

    citation_by_source = {
        (source["raw_document"], source["chunk"]): source["citation"]
        for source in sources
    }
    trace = dict(trace_payload)
    trace["important_evidence"] = [
        {
            **item,
            "citation": citation_by_source.get((item.get("document"), item.get("chunk"))),
        }
        for item in trace.get("important_evidence", [])
    ]
    return compact_trace(trace, category)


def compact_trace(trace: dict, category: str) -> dict:
    query_type = trace.get("query_type")
    return {
        "query_type": query_type,
        "category": category,
        "route_reason": route_reason(query_type),
        "important_evidence": [
            {"citation": item.get("citation")}
            for item in trace.get("important_evidence", [])
            if item.get("citation")
        ],
        "conclusion": {
            "answer_type": trace.get("conclusion", {}).get("answer_type"),
        },
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = LegalRAG()
    try:
        yield
    finally:
        app.state.rag.close()


app = FastAPI(lifespan=lifespan)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    strategy: Literal["vector", "graph", "neuro_symbolic"] = "graph"
    category: str = "auto"


def get_rag(request: Request) -> LegalRAG:
    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system is starting.")
    return rag


@app.get("/")
def home() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "frontend" / "index.html")


@app.get("/documents")
def list_documents(rag_system: Annotated[LegalRAG, Depends(get_rag)]) -> dict:
    return {"documents": rag_system.list_documents()}


@app.get("/documents/raw/{document_path:path}")
def raw_document(document_path: str) -> FileResponse:
    path = (RAW_DOCUMENTS_PATH / document_path).resolve()
    try:
        path.relative_to(RAW_DOCUMENTS_PATH)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="Document not found.") from error

    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="Document not found.")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=path.name,
        content_disposition_type="inline",
    )


@app.post("/ask")
async def ask(
    request: Question,
    rag_system: Annotated[LegalRAG, Depends(get_rag)],
) -> StreamingResponse:
    try:
        chunks = rag_system.retrieve(
            request.question, strategy=request.strategy, category=request.category
        )
    except RetrievalUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    async def stream():
        trace_payload = next(
            (
                chunk.meta.get("symbolic_trace")
                for chunk in chunks
                if chunk.meta.get("symbolic_trace")
            ),
            None,
        )
        source_chunks = chunks
        sources = [
            {
                "document": chunk.meta.get("document_name", ""),
                "raw_document": chunk.meta.get("document_name", ""),
                "pdf_url": document_pdf_url(chunk.meta.get("source_path")),
                "citation": citation_number,
                "chunk": chunk.meta.get("chunk_number"),
                "score": chunk.score,
                "text": chunk.content,
                "heading_path": chunk.meta.get("heading_path"),
                "article": chunk.meta.get("article_number"),
                "article_title": chunk.meta.get("article_title"),
                "legal_unit_kind": chunk.meta.get("legal_unit_kind"),
                "legal_unit_number": chunk.meta.get("legal_unit_number"),
                "legal_unit_title": chunk.meta.get("legal_unit_title"),
                "specific_citation": chunk.meta.get("specific_citation"),
            }
            for citation_number, chunk in enumerate(source_chunks, start=1)
        ]
        graph_facts = graph_fact_rows(chunks)
        trace_payload = add_trace_citations(trace_payload, sources, request.category)
        yield json.dumps({"type": "sources", "sources": sources}) + "\n"
        if graph_facts:
            yield json.dumps({"type": "graph_facts", "graph_facts": graph_facts}) + "\n"
        if trace_payload:
            yield json.dumps({"type": "trace", "trace": trace_payload}) + "\n"

        tokens = asyncio.Queue()

        async def send_token(token: str) -> None:
            await tokens.put(token)

        async def generate() -> None:
            try:
                await rag_system.generate_async(
                    request.question,
                    source_chunks,
                    send_token,
                )
            except Exception as error:
                await tokens.put(error)
            finally:
                await tokens.put(None)

        generation = asyncio.create_task(generate())

        try:
            while True:
                token = await tokens.get()
                if token is None:
                    break
                if isinstance(token, Exception):
                    yield json.dumps({"type": "error"}) + "\n"
                    await generation
                    return
                yield json.dumps({"type": "token", "text": token}) + "\n"

            await generation
            yield json.dumps({"type": "done"}) + "\n"
        finally:
            if not generation.done():
                generation.cancel()
                with suppress(asyncio.CancelledError):
                    await generation

    return StreamingResponse(stream(), media_type="application/x-ndjson")
