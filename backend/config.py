import os
from pathlib import Path


def env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def configure_runtime_environment() -> None:
    project_root = Path(__file__).resolve().parent.parent
    os.environ.setdefault("HF_HOME", str(project_root / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


EMBEDDING_MODEL = os.getenv(
    "LEGALRAG_EMBEDDING_MODEL",
    "intfloat/multilingual-e5-small",
)
EMBEDDING_PROFILES = {
    "BAAI/bge-m3": {
        "max_words": "260",
        "top_k": "4",
        "batch_size": "16",
    },
    "intfloat/multilingual-e5-small": {
        "max_words": "140",
        "top_k": "6",
        "batch_size": "64",
    },
}
EMBEDDING_PROFILE = EMBEDDING_PROFILES.get(EMBEDDING_MODEL, {})
GENERATION_MODEL = os.getenv("LEGALRAG_GENERATION_MODEL", "qwen3.5:4b")
JUDGE_MODEL = os.getenv("LEGALRAG_JUDGE_MODEL", "qwen2.5:1.5b-instruct")
LLM_PROVIDER = os.getenv("LEGALRAG_LLM_PROVIDER", "ollama").lower()
OLLAMA_URL = os.getenv("LEGALRAG_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_KEEP_ALIVE = os.getenv("LEGALRAG_OLLAMA_KEEP_ALIVE", "1m")
VLLM_URL = os.getenv("LEGALRAG_VLLM_URL", "http://127.0.0.1:8001")

PROXY_SHARED_SECRET = os.getenv("LEGALRAG_PROXY_SECRET", "")

NEO4J_URI = os.getenv("LEGALRAG_NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("LEGALRAG_NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("LEGALRAG_NEO4J_PASSWORD", "legalrag-password")
NEO4J_DATABASE = os.getenv("LEGALRAG_NEO4J_DATABASE", "neo4j")
NEO4J_VECTOR_INDEX = os.getenv(
    "LEGALRAG_NEO4J_VECTOR_INDEX",
    "chunk_embedding_index",
)

TOP_K = int(os.getenv("LEGALRAG_TOP_K", EMBEDDING_PROFILE.get("top_k", "4")))
RETRIEVAL_STRATEGIES = ("vector", "graph", "neuro_symbolic")
RETRIEVAL_STRATEGY_ALIASES = {"hybrid": "graph"}
EMBEDDING_BATCH_SIZE = int(
    os.getenv(
        "LEGALRAG_EMBEDDING_BATCH_SIZE",
        EMBEDDING_PROFILE.get("batch_size", "32"),
    )
)
TEMPERATURE = float(os.getenv("LEGALRAG_TEMPERATURE", "0.1"))
CONTEXT_WINDOW = int(os.getenv("LEGALRAG_CONTEXT_WINDOW", "4096"))
MAX_OUTPUT_TOKENS = int(os.getenv("LEGALRAG_MAX_OUTPUT_TOKENS", "700"))
OLLAMA_THINK = env_bool("LEGALRAG_OLLAMA_THINK", "false")

DEFAULT_MAX_WORDS = int(
    os.getenv("LEGALRAG_MAX_WORDS", EMBEDDING_PROFILE.get("max_words", "180"))
)
# Bump this whenever chunk text or indexed graph metadata changes.
CHUNKING_VERSION = "word-block-v18-structured-legal-body"

if DEFAULT_MAX_WORDS <= 0:
    raise ValueError("LEGALRAG_MAX_WORDS must be greater than 0.")


def normalize_retrieval_strategy(strategy: str) -> str:
    return RETRIEVAL_STRATEGY_ALIASES.get(strategy, strategy)
