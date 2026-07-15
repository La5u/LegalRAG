import hashlib
import json
from pathlib import Path

from backend.catalog import PROJECT_ROOT
from backend.config import CHUNKING_VERSION
from backend.document import Document


CACHE_DIR = PROJECT_ROOT / ".cache" / "legalrag" / "chunks"


def _cache_path(source_path: str) -> Path:
    name = hashlib.sha256(source_path.encode()).hexdigest()[:24]
    return CACHE_DIR / f"{name}.json"


def load_cached_chunks(source_path: str, source_mtime_ns: int) -> list[Document] | None:
    path = _cache_path(source_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        data.get("source_path") != source_path
        or data.get("source_mtime_ns") != source_mtime_ns
        or data.get("chunking_version") != CHUNKING_VERSION
    ):
        return None
    return [Document(row["content"], row["meta"]) for row in data["chunks"]]


def save_cached_chunks(
    source_path: str, source_mtime_ns: int, chunks: list[Document],
) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(source_path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "source_path": source_path,
                "source_mtime_ns": source_mtime_ns,
                "chunking_version": CHUNKING_VERSION,
                "chunks": [
                    {"content": chunk.content, "meta": chunk.meta}
                    for chunk in chunks
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
