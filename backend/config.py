import os
from pathlib import Path


def env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


EMBEDDING_MODEL = os.getenv(
    "LEGALRAG_EMBEDDING_MODEL",
    "intfloat/multilingual-e5-small",
)
EMBEDDING_PROFILES = {
    "BAAI/bge-m3": {
        "max_words": "260",
        "top_k": "4",
        "paragraph_neighbors": "1",
        "batch_size": "16",
    },
    "intfloat/multilingual-e5-small": {
        "max_words": "140",
        "top_k": "6",
        "paragraph_neighbors": "1",
        "batch_size": "64",
    },
}
EMBEDDING_PROFILE = EMBEDDING_PROFILES.get(EMBEDDING_MODEL, {})
GENERATION_MODEL = os.getenv("LEGALRAG_GENERATION_MODEL", "qwen3.5:4b")
OLLAMA_URL = os.getenv("LEGALRAG_OLLAMA_URL", "http://127.0.0.1:11434")

NEO4J_URI = os.getenv("LEGALRAG_NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("LEGALRAG_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("LEGALRAG_NEO4J_PASSWORD", "legalrag-password")
NEO4J_DATABASE = os.getenv("LEGALRAG_NEO4J_DATABASE", "neo4j")
NEO4J_VECTOR_INDEX = os.getenv(
    "LEGALRAG_NEO4J_VECTOR_INDEX",
    "chunk_embedding_index",
)

TOP_K = int(os.getenv("LEGALRAG_TOP_K", EMBEDDING_PROFILE.get("top_k", "4")))
PARAGRAPH_NEIGHBORS = int(
    os.getenv(
        "LEGALRAG_PARAGRAPH_NEIGHBORS",
        EMBEDDING_PROFILE.get("paragraph_neighbors", "1"),
    )
)
EMBEDDING_BATCH_SIZE = int(
    os.getenv(
        "LEGALRAG_EMBEDDING_BATCH_SIZE",
        EMBEDDING_PROFILE.get("batch_size", "32"),
    )
)
TEMPERATURE = float(os.getenv("LEGALRAG_TEMPERATURE", "0.1"))
CONTEXT_WINDOW = int(os.getenv("LEGALRAG_CONTEXT_WINDOW", "4096"))
MAX_OUTPUT_TOKENS = int(os.getenv("LEGALRAG_MAX_OUTPUT_TOKENS", "300"))
OLLAMA_THINK = env_bool("LEGALRAG_OLLAMA_THINK", "false")

DEFAULT_MAX_WORDS = int(
    os.getenv("LEGALRAG_MAX_WORDS", EMBEDDING_PROFILE.get("max_words", "180"))
)
# Bump this whenever chunk text or indexed graph metadata changes.
CHUNKING_VERSION = "word-block-v11"

_RUNTIME_ENVIRONMENT_CONFIGURED = False

if DEFAULT_MAX_WORDS <= 0:
    raise ValueError("LEGALRAG_MAX_WORDS must be greater than 0.")


def configure_runtime_environment() -> None:
    global _RUNTIME_ENVIRONMENT_CONFIGURED
    if _RUNTIME_ENVIRONMENT_CONFIGURED:
        return

    project_root = Path(__file__).resolve().parent.parent
    os.environ["HF_HOME"] = str(project_root / ".cache" / "huggingface")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    _RUNTIME_ENVIRONMENT_CONFIGURED = True
