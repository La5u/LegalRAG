import asyncio
import json
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.document_types import document_type_from_source_path
from backend.rag import LegalRAG
from backend.retrieval import RetrievalUnavailable

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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


def graph_fact_rows(chunks) -> list[dict]:
    rows = []
    for chunk in chunks:
        for fact in chunk.meta.get("graph_facts", []):
            rows.append(fact.to_response())
    return rows


def add_trace_citations(trace_payload: dict | None, sources: list[dict]) -> dict | None:
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
    return compact_trace(trace)


def compact_trace(trace: dict) -> dict:
    return {
        "query_type": trace.get("query_type"),
        "summary": trace.get("summary", []),
        "important_evidence": trace.get("important_evidence", []),
        "conclusion": trace.get("conclusion", {}),
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
                "citation": citation_number,
                "chunk": chunk.meta.get("chunk_number"),
                "score": chunk.score,
                "text": chunk.content,
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
        trace_payload = add_trace_citations(trace_payload, sources)
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
