import os
import hashlib
from pathlib import Path


def configure_runtime_environment() -> None:
    project_root = Path(__file__).resolve().parent.parent
    os.environ.setdefault("HF_HOME", str(project_root / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")


EMBEDDING_MODEL = os.getenv(
    "LEGALRAG_EMBEDDING_MODEL",
    "intfloat/multilingual-e5-small",
)
EMBEDDING_PROFILES = {
    "BAAI/bge-m3": {
        "max_words": "260",
        "top_k": "4",
        "batch_size": "16",
        "query_prefix": "",
        "passage_prefix": "",
    },
    "intfloat/multilingual-e5-small": {
        "max_words": "140",
        "top_k": "6",
        "batch_size": "64",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
}
EMBEDDING_PROFILE = EMBEDDING_PROFILES.get(EMBEDDING_MODEL, {})
EMBEDDING_QUERY_PREFIX = EMBEDDING_PROFILE.get("query_prefix", "")
EMBEDDING_PASSAGE_PREFIX = EMBEDDING_PROFILE.get("passage_prefix", "")
GENERATION_MODEL = os.getenv("LEGALRAG_GENERATION_MODEL", "qwen3.5:4b")
JUDGE_MODEL = os.getenv("LEGALRAG_JUDGE_MODEL", "qwen2.5:1.5b-instruct")
LLM_PROVIDER = os.getenv("LEGALRAG_LLM_PROVIDER", "ollama").lower()
OLLAMA_URL = os.getenv("LEGALRAG_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_KEEP_ALIVE = os.getenv("LEGALRAG_OLLAMA_KEEP_ALIVE", "1m")
VLLM_URL = os.getenv("LEGALRAG_VLLM_URL", "http://127.0.0.1:8001")
LLM_BASE_URL = os.getenv("LEGALRAG_LLM_BASE_URL", "").rstrip("/")
LLM_API_KEY = os.getenv("LEGALRAG_LLM_API_KEY", "")

NEO4J_URI = os.getenv("LEGALRAG_NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("LEGALRAG_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("LEGALRAG_NEO4J_PASSWORD", "legalrag-password")
NEO4J_DATABASE = os.getenv("LEGALRAG_NEO4J_DATABASE", "neo4j")
NEO4J_VECTOR_INDEX = os.getenv(
    "LEGALRAG_NEO4J_VECTOR_INDEX",
    "chunk_embedding_index",
)

TOP_K = int(os.getenv("LEGALRAG_TOP_K", EMBEDDING_PROFILE.get("top_k", "4")))
# This is a word-based approximation of the prompt budget.  It keeps every
# retrieval mode within the same evidence limit before the answer is generated.
MAX_CONTEXT_WORDS = int(os.getenv("LEGALRAG_MAX_CONTEXT_WORDS", "1200"))
# The number of source chunks shown to the model and user after final evidence
# selection. Retrieval can still inspect more candidates before this limit.
MAX_EVIDENCE_CHUNKS = int(os.getenv("LEGALRAG_MAX_EVIDENCE_CHUNKS", "4"))
RETRIEVAL_STRATEGIES = ("vector", "graph", "neuro_symbolic")
RETRIEVAL_STRATEGY_ALIASES = {"hybrid": "graph"}
EMBEDDING_BATCH_SIZE = int(
    os.getenv(
        "LEGALRAG_EMBEDDING_BATCH_SIZE",
        EMBEDDING_PROFILE.get("batch_size", "32"),
    )
)
# Deterministic generation makes retrieval-mode comparisons reproducible.
TEMPERATURE = float(os.getenv("LEGALRAG_TEMPERATURE", "0"))
GENERATION_SEED = int(os.getenv("LEGALRAG_GENERATION_SEED", "0"))
CONTEXT_WINDOW = int(os.getenv("LEGALRAG_CONTEXT_WINDOW", "4096"))
MAX_OUTPUT_TOKENS = int(os.getenv("LEGALRAG_MAX_OUTPUT_TOKENS", "700"))
OLLAMA_THINK = os.getenv("LEGALRAG_OLLAMA_THINK", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

DEFAULT_MAX_WORDS = int(
    os.getenv("LEGALRAG_MAX_WORDS", EMBEDDING_PROFILE.get("max_words", "180"))
)
PROSE_MAX_WORDS = int(os.getenv("LEGALRAG_PROSE_MAX_WORDS", "220"))


def pipeline_fingerprint(names: tuple[str, ...]) -> str:
    """Hash one indexing stage so unrelated changes retain reusable work."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in names:
        digest.update((root / name).read_bytes())
    return digest.hexdigest()[:16]


CHUNKING_VERSION = "chunks-" + pipeline_fingerprint((
    "catalog.py", "chunker.py", "loaders.py",
)) + f"-{DEFAULT_MAX_WORDS}-{PROSE_MAX_WORDS}"
INDEXING_VERSION = "graph-" + pipeline_fingerprint((
    "connections.py", "definitions.py", "graph_index.py", "indexing.py",
    "triples.py",
))


def chunk_word_limit(document_type: str | None) -> int:
    if document_type in {"guidance", "qna", "precedent"}:
        return PROSE_MAX_WORDS
    return DEFAULT_MAX_WORDS


if DEFAULT_MAX_WORDS <= 0:
    raise ValueError("LEGALRAG_MAX_WORDS must be greater than 0.")
if PROSE_MAX_WORDS <= 0:
    raise ValueError("LEGALRAG_PROSE_MAX_WORDS must be greater than 0.")
if MAX_CONTEXT_WORDS <= 0:
    raise ValueError("LEGALRAG_MAX_CONTEXT_WORDS must be greater than 0.")
if MAX_EVIDENCE_CHUNKS <= 0:
    raise ValueError("LEGALRAG_MAX_EVIDENCE_CHUNKS must be greater than 0.")


def normalize_retrieval_strategy(strategy: str) -> str:
    return RETRIEVAL_STRATEGY_ALIASES.get(strategy, strategy)
