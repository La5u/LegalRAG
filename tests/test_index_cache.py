from backend.document import Document
from backend import index_cache


def test_chunk_cache_round_trip_and_mtime_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(index_cache, "CACHE_DIR", tmp_path)
    chunks = [Document("내용", {"chunk_number": 1})]

    index_cache.save_cached_chunks("documents/raw/test.hwpx", 123, chunks)

    cached = index_cache.load_cached_chunks("documents/raw/test.hwpx", 123)
    assert cached == chunks
    assert index_cache.load_cached_chunks("documents/raw/test.hwpx", 124) is None
