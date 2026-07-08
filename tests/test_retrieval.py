import pytest

from backend.config import NEO4J_VECTOR_INDEX
from backend.document import Document
from backend.graph_retrieve import GraphFact
from backend.records import chunk_meta
from backend.retrieval import (
    RetrievalUnavailable,
    attach_graph_facts,
    attach_legal_context,
    attach_symbolic_trace,
    expand_legal_units,
    filter_by_category,
    graph_facts_from_meta,
    merge_documents,
    merge_ranked_expansion,
    quoted_terms,
    retrieve_case_source_matches,
    retrieve_contract_matches,
    retrieve_document_summary_matches,
    retrieve_exact_phrase_matches,
    retrieve_graphrag,
    retrieve_legal_unit_matches,
    retrieve_source_constrained_matches,
    retrieve_sparse_chunks,
    retrieve_chunks,
    retrieve_unconstrained,
    retrieve_vector_chunks,
    retrieve_vectorrag,
    should_expand_legal_units,
    specific_legal_citation,
)
from backend.routing import QueryRoute, classify_query
from backend.sources import SourceMatcher


# ---------------------------------------------------------------------------
# should_expand_legal_units — detect article/unit references in query
# ---------------------------------------------------------------------------
class TestShouldExpandLegalUnits:
    @pytest.mark.parametrize(
        "query, expected",
        [
            ("제15조 내용을 알려주세요", True),
            ("제1항에 해당하는 경우", True),
            ("제2호의 요건", True),
            ("가목의 규정", True),
            ("조문 전체를 보여주세요", True),
            ("어느 조에 해당하나요", True),
            ("Article 15", True),
            ("Section 3", True),
            ("개인정보 처리 방침", False),
            ("통지 의무", False),
        ],
    )
    def test_detects_unit_references(self, query, expected):
        assert should_expand_legal_units(query) == expected


# ---------------------------------------------------------------------------
# quoted_terms — extract quoted strings
# ---------------------------------------------------------------------------
class TestQuotedTerms:
    def test_extracts_double_quotes(self):
        terms = quoted_terms('이 법에서 "개인정보"란 무엇인가요')
        assert terms == ["개인정보"]

    def test_extracts_single_quotes(self):
        terms = quoted_terms("이 법에서 '개인정보'란 무엇인가요")
        assert terms == ["개인정보"]

    def test_extracts_korean_quotes(self):
        terms = quoted_terms('이 법에서 "개인정보"란 무엇인가요')
        assert terms == ["개인정보"]

    def test_multiple_quotes(self):
        terms = quoted_terms('"개인정보"와 "정보주체"의 관계')
        assert terms == ["개인정보", "정보주체"]

    def test_no_quotes(self):
        assert quoted_terms("개인정보 처리 방침") == []

    def test_empty_quotes(self):
        assert quoted_terms('"" ""') == []


# ---------------------------------------------------------------------------
# specific_legal_citation — build citation string from metadata
# ---------------------------------------------------------------------------
class TestSpecificLegalCitation:
    def test_article_only(self):
        meta = {"article_number": "제15조"}
        assert specific_legal_citation(meta, "some content") == "제15조"

    def test_article_with_circled_clause(self):
        meta = {"article_number": "제15조", "legal_unit_kind": "clause", "legal_unit_number": "①"}
        content = "① 개인정보의 수집·이용"
        assert specific_legal_citation(meta, content) == "제15조제1항"

    def test_article_with_item(self):
        meta = {"article_number": "제15조", "legal_unit_kind": "item", "legal_unit_number": "1."}
        assert specific_legal_citation(meta, "some content") == "제15조제1호"

    def test_article_with_subitem(self):
        meta = {"article_number": "제15조", "legal_unit_kind": "subitem", "legal_unit_number": "가."}
        assert specific_legal_citation(meta, "some content") == "제15조가목"

    def test_article_kind_returns_article(self):
        meta = {"article_number": "제15조", "legal_unit_kind": "article", "legal_unit_number": "제15조"}
        assert specific_legal_citation(meta, "some content") == "제15조"

    def test_no_article_returns_none(self):
        assert specific_legal_citation({}, "some content") is None

    def test_circled_in_content_overrides_unit(self):
        meta = {"article_number": "제15조", "legal_unit_kind": "article", "legal_unit_number": "제15조"}
        content = "② 정보주체의 권리"
        assert specific_legal_citation(meta, content) == "제15조제2항"


# ---------------------------------------------------------------------------
# merge_documents — dedup by id
# ---------------------------------------------------------------------------
class TestMergeDocuments:
    def test_deduplicates_by_id(self):
        docs = [
            Document("a", {"id": "1"}),
            Document("b", {"id": "2"}),
            Document("c", {"id": "1"}),
        ]
        result = merge_documents(docs)
        assert len(result) == 2
        assert result[0].content == "a"
        assert result[1].content == "b"

    def test_keeps_docs_without_id(self):
        docs = [
            Document("a", {}),
            Document("b", {}),
        ]
        result = merge_documents(docs)
        assert len(result) == 2

    def test_mixed_id_and_no_id(self):
        docs = [
            Document("a", {"id": "1"}),
            Document("b", {}),
            Document("c", {"id": "1"}),
        ]
        result = merge_documents(docs)
        # "a" kept (id "1"), "b" kept (no id, always kept), "c" dropped (id "1" dup)
        assert len(result) == 2
        assert result[0].content == "a"
        assert result[1].content == "b"


# ---------------------------------------------------------------------------
# filter_by_category — filter by document type
# ---------------------------------------------------------------------------
class TestFilterByCategory:
    def _make_doc(self, source_path):
        return Document("content", {"source_path": source_path})

    def test_filters_by_statute(self):
        docs = [
            self._make_doc("documents/raw/statutes/법.pdf"),
            self._make_doc("documents/raw/precedents/판례.pdf"),
        ]
        result = filter_by_category(docs, "statute")
        assert len(result) == 1
        assert "statutes" in result[0].meta["source_path"]

    def test_all_category_returns_all(self):
        docs = [
            self._make_doc("documents/raw/statutes/법.pdf"),
            self._make_doc("documents/raw/precedents/판례.pdf"),
        ]
        result = filter_by_category(docs, "all")
        assert len(result) == 2

    def test_empty_category_returns_all(self):
        docs = [
            self._make_doc("documents/raw/statutes/법.pdf"),
        ]
        result = filter_by_category(docs, "")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# smoke.py — FakeEmbeddingModel, smoke_chunk
# ---------------------------------------------------------------------------
class TestFakeEmbeddingModel:
    def test_single_text(self):
        from scripts.smoke import FakeEmbeddingModel

        model = FakeEmbeddingModel(16)
        result = model.encode("test")
        assert len(result) == 16
        assert all(v == 0.0 for v in result)

    def test_batch_texts(self):
        from scripts.smoke import FakeEmbeddingModel

        model = FakeEmbeddingModel(8)
        result = model.encode(["a", "b", "c"])
        assert len(result) == 3
        assert all(len(vec) == 8 for vec in result)

    def test_custom_dimensions(self):
        from scripts.smoke import FakeEmbeddingModel

        model = FakeEmbeddingModel(32)
        assert len(model.encode("test")) == 32


class TestSmokeChunk:
    def test_creates_document(self):
        from scripts.smoke import smoke_chunk

        doc = smoke_chunk(1, "제1조(목적) 이 법의 목적은.", "제1조", "목적", 1)
        assert doc.content == "제1조(목적) 이 법의 목적은."
        assert doc.meta["chunk_number"] == 1
        assert doc.meta["article_number"] == "제1조"
        assert doc.meta["article_title"] == "목적"
        assert doc.meta["chapter_number"] == "제1장"

    def test_group_id_is_chunk_minus_one(self):
        from scripts.smoke import smoke_chunk

        doc = smoke_chunk(3, "content", "제2조", "정의", 2)
        assert doc.meta["group_id"] == 2


# ---------------------------------------------------------------------------
# Fakes for the retrieval pipeline (no live Neo4j or model needed)
# ---------------------------------------------------------------------------
# sentence-transformers returns a numpy array whose .tolist() gives a list of
# floats. _Embedding mimics that minimal contract so retrieve_vector_chunks can
# call .encode(...).tolist() without the real model.
class _Embedding(list):
    def tolist(self):
        return list(self)


class FakeEmbedder:
    def __init__(self, dimensions=4):
        self.dimensions = dimensions

    def encode(self, texts, **_kwargs):
        if isinstance(texts, str):
            return _Embedding([0.0] * self.dimensions)
        return [_Embedding([0.0] * self.dimensions) for _ in texts]


class FakeNeo4j:
    """Stand-in for the run_read callable passed through the retrieval layer.

    Dispatches by query signature so a single canned set of chunk records can
    feed every retriever: vector-index existence, vector search, legal-context
    lookup, graph-fact queries (which return no facts), and generic chunk reads."""

    def __init__(self, chunks=None, context=None, vector_index_exists=True):
        self._chunks = chunks or []
        self._context = context or []
        self.vector_index_exists = vector_index_exists
        self.calls = []

    def run_read(self, query, **params):
        self.calls.append(query)
        if "SHOW VECTOR INDEXES" in query:
            return [{"name": NEO4J_VECTOR_INDEX}] if self.vector_index_exists else []
        if "Concept" in query or "Holding" in query or "Term" in query:
            return []
        if "db.index.vector.queryNodes" in query:
            return list(self._chunks)
        if "chunk.id IN $chunk_ids" in query:
            return list(self._context)
        return list(self._chunks)


class FakeSourceMatcher:
    def __init__(self, source_paths):
        self._paths = list(source_paths)

    def constraints(self, _query):
        return list(self._paths)


def chunk_record(num=1, content="chunk content", source_path="documents/raw/statutes/법.pdf", **extra):
    return {
        "id": f"chunk-{num}",
        "content": content,
        "heading_path": "제1장 총칙 > 제1조 목적",
        "chunk_number": num,
        "document_name": "법.pdf",
        "group_id": num - 1,
        "part": 0,
        "source_path": source_path,
        "score": round(1.0 - num * 0.1, 3),
        **extra,
    }


def _graph_fact(score=0.5, path="개인정보처리자 -MUST_NOTIFY-> 정보주체", sentence="통지하여야 한다"):
    return GraphFact(
        id="gf-1",
        document="법.pdf",
        source_path="documents/raw/statutes/법.pdf",
        chunk="graph-1",
        relation="MUST_NOTIFY",
        path=path,
        sentence=sentence,
        score=score,
    )


# ---------------------------------------------------------------------------
# retrieve_exact_phrase_matches
# ---------------------------------------------------------------------------
class TestRetrieveExactPhraseMatches:
    def test_returns_documents_and_marks_match(self):
        db = FakeNeo4j(chunks=[chunk_record(1, content="개인정보를 보호한다")])
        docs = retrieve_exact_phrase_matches(db.run_read, '"개인정보" 보호')
        assert len(docs) == 1
        assert docs[0].meta["exact_phrase_match"] is True
        assert docs[0].meta["matched_phrases"] == ["개인정보"]

    def test_no_phrases_returns_empty_without_query(self):
        db = FakeNeo4j(chunks=[chunk_record(1)])
        assert retrieve_exact_phrase_matches(db.run_read, "개인정보 보호") == []
        assert db.calls == []


# ---------------------------------------------------------------------------
# retrieve_vector_chunks
# ---------------------------------------------------------------------------
class TestRetrieveVectorChunks:
    def test_returns_documents_with_scores(self):
        db = FakeNeo4j(chunks=[chunk_record(1), chunk_record(2)])
        docs = retrieve_vector_chunks(db.run_read, FakeEmbedder(), top_k=4, query="개인정보")
        assert [d.meta["id"] for d in docs] == ["chunk-1", "chunk-2"]
        assert docs[0].score == 0.9
        assert docs[1].score == 0.8

    def test_raises_when_no_index(self):
        db = FakeNeo4j(chunks=[chunk_record(1)], vector_index_exists=False)
        with pytest.raises(RetrievalUnavailable):
            retrieve_vector_chunks(db.run_read, FakeEmbedder(), top_k=4, query="개인정보")


# ---------------------------------------------------------------------------
# retrieve_sparse_chunks
# ---------------------------------------------------------------------------
class TestRetrieveSparseChunks:
    def test_returns_documents(self):
        db = FakeNeo4j(chunks=[chunk_record(1)])
        route = QueryRoute(kind="general", search_terms=["개인정보", "처리"])
        docs = retrieve_sparse_chunks(db.run_read, route)
        assert len(docs) == 1

    def test_no_terms_returns_empty_without_query(self):
        db = FakeNeo4j(chunks=[chunk_record(1)])
        route = QueryRoute(kind="general", search_terms=[])
        assert retrieve_sparse_chunks(db.run_read, route) == []
        assert db.calls == []


# ---------------------------------------------------------------------------
# retrieve_legal_unit_matches
# ---------------------------------------------------------------------------
class TestRetrieveLegalUnitMatches:
    def test_returns_documents(self):
        db = FakeNeo4j(chunks=[chunk_record(1)])
        route = QueryRoute(kind="general", search_terms=["개인정보"])
        docs = retrieve_legal_unit_matches(db.run_read, route)
        assert len(docs) == 1

    def test_no_terms_or_articles_returns_empty_without_query(self):
        db = FakeNeo4j(chunks=[chunk_record(1)])
        route = QueryRoute(kind="general", search_terms=[], article_numbers=[])
        assert retrieve_legal_unit_matches(db.run_read, route) == []
        assert db.calls == []


# ---------------------------------------------------------------------------
# retrieve_source_constrained_matches / summary / contract / case
# ---------------------------------------------------------------------------
class TestRetrieveSourceConstrainedMatches:
    def test_returns_documents(self):
        db = FakeNeo4j(chunks=[chunk_record(1)])
        route = QueryRoute(kind="general", search_terms=["개인정보"])
        docs = retrieve_source_constrained_matches(
            db.run_read, "개인정보 보호", route, ["documents/raw/statutes/법.pdf"]
        )
        assert len(docs) == 1


class TestRetrieveDocumentSummaryMatches:
    def test_returns_documents(self):
        db = FakeNeo4j(chunks=[chunk_record(1)])
        docs = retrieve_document_summary_matches(db.run_read, "주요 내용", ["documents/raw/statutes/법.pdf"])
        assert len(docs) == 1


class TestRetrieveContractMatches:
    def test_returns_documents(self):
        db = FakeNeo4j(chunks=[chunk_record(1, content="임대인의 의무")])
        docs = retrieve_contract_matches(db.run_read, "임대인 의무", ["documents/raw/contracts/계약서.pdf"])
        assert len(docs) == 1


class TestRetrieveCaseSourceMatches:
    def test_returns_documents(self):
        db = FakeNeo4j(chunks=[chunk_record(1, content="판단 이유")])
        docs = retrieve_case_source_matches(db.run_read, "판례 내용", ["documents/raw/precedents/판례.pdf"])
        assert len(docs) == 1


# ---------------------------------------------------------------------------
# attach_legal_context
# ---------------------------------------------------------------------------
class TestAttachLegalContext:
    def test_enriches_meta_with_legal_unit_and_citation(self):
        doc = Document("① 내용", {"id": "chunk-1", "chunk_number": 1, "source_path": "documents/raw/statutes/법.pdf"})
        context = [{
            "id": "chunk-1",
            "legal_unit_id": "lu-1",
            "legal_unit_kind": "clause",
            "legal_unit_number": "①",
            "legal_unit_title": None,
            "article_id": "a-1",
            "article_number": "제15조",
            "article_title": "목적",
        }]
        db = FakeNeo4j(context=context)
        result = attach_legal_context(db.run_read, [doc])
        assert result[0].meta["legal_unit_kind"] == "clause"
        assert result[0].meta["article_number"] == "제15조"
        assert result[0].meta["specific_citation"] == "제15조제1항"
        assert result[0].meta["document_type"] == "statute"

    def test_no_ids_returns_unchanged_without_query(self):
        db = FakeNeo4j(context=[{"id": "x"}])
        doc = Document("내용", {"chunk_number": 1})
        result = attach_legal_context(db.run_read, [doc])
        assert len(result) == 1
        assert result[0].content == "내용"
        assert db.calls == []


# ---------------------------------------------------------------------------
# expand_legal_units
# ---------------------------------------------------------------------------
class TestExpandLegalUnits:
    def test_no_ids_returns_unchanged_without_query(self):
        db = FakeNeo4j(chunks=[chunk_record(2)])
        seed = Document("seed", {"chunk_number": 1})
        route = QueryRoute(kind="general")
        result = expand_legal_units(db.run_read, [seed], route, [])
        assert result == [seed]
        assert db.calls == []

    def test_merges_expansion_records_by_rank(self):
        seed = Document("seed", {"id": "chunk-1", "chunk_number": 1})
        expansion = [{**chunk_record(2), "rank": 0}, {**chunk_record(3), "rank": 0}]
        run_read = lambda query, **params: expansion
        route = QueryRoute(kind="general")
        result = expand_legal_units(run_read, [seed], route, [])
        assert len(result) == 3
        assert result[0].content == "seed"
        assert {d.meta["id"] for d in result} == {"chunk-1", "chunk-2", "chunk-3"}


# ---------------------------------------------------------------------------
# merge_ranked_expansion (pure)
# ---------------------------------------------------------------------------
class TestMergeRankedExpansion:
    def _doc(self, doc_id):
        return Document(f"c-{doc_id}", {"id": doc_id})

    def _record(self, doc_id, rank):
        return {
            "id": doc_id,
            "content": f"c-{doc_id}",
            "heading_path": None,
            "chunk_number": int(doc_id),
            "document_name": "d",
            "group_id": 0,
            "part": 0,
            "source_path": "p",
            "score": 0.5,
            "rank": rank,
        }

    def test_interleaves_records_by_rank(self):
        retrieved = [self._doc("1"), self._doc("2")]
        records = [self._record("3", 0), self._record("4", 1)]
        result = merge_ranked_expansion(retrieved, records)
        assert [d.meta["id"] for d in result] == ["1", "3", "2", "4"]

    def test_dedups_repeated_ids(self):
        retrieved = [self._doc("1")]
        records = [self._record("1", 0)]
        result = merge_ranked_expansion(retrieved, records)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# attach_graph_facts (pure)
# ---------------------------------------------------------------------------
class TestAttachGraphFacts:
    def _chunks(self):
        return [Document("a", {"id": "1"}), Document("b", {"id": "2"})]

    def test_attaches_sorted_facts_to_first_chunk(self):
        facts = [_graph_fact(score=0.5, path="P1"), _graph_fact(score=0.9, path="P2")]
        result = attach_graph_facts(self._chunks(), facts, top_k=2)
        assert len(result) == 2
        assert [f.score for f in result[0].meta["graph_facts"]] == [0.9, 0.5]
        assert "graph_facts" not in result[1].meta

    def test_dedups_by_key(self):
        facts = [_graph_fact(score=0.5, path="P"), _graph_fact(score=0.9, path="P")]
        result = attach_graph_facts(self._chunks(), facts, top_k=2)
        assert len(result[0].meta["graph_facts"]) == 1
        assert result[0].meta["graph_facts"][0].score == 0.9

    def test_no_facts_returns_chunks_unchanged(self):
        result = attach_graph_facts(self._chunks(), [], top_k=2)
        assert "graph_facts" not in result[0].meta

    def test_empty_chunks_returns_empty(self):
        assert attach_graph_facts([], [_graph_fact()], top_k=2) == []


# ---------------------------------------------------------------------------
# graph_facts_from_meta (reads graph_facts from chunk meta)
# ---------------------------------------------------------------------------
class TestGraphFactDocumentsFromMeta:
    def test_converts_meta_facts_to_documents(self):
        chunk = Document("a", {"id": "1", "graph_facts": [_graph_fact()]})
        result = graph_facts_from_meta([chunk])
        assert len(result) == 1
        assert "-MUST_NOTIFY->" in result[0].content

    def test_no_facts_returns_empty(self):
        chunk = Document("a", {"id": "1"})
        assert graph_facts_from_meta([chunk]) == []


# ---------------------------------------------------------------------------
# attach_symbolic_trace
# ---------------------------------------------------------------------------
class TestAttachSymbolicTrace:
    def test_sets_trace_on_first_chunk(self):
        evidence = [
            Document("개인정보를 보호한다", {"id": "1"}),
            Document("기타 내용", {"id": "2"}),
        ]
        result = attach_symbolic_trace("개인정보 보호", evidence)
        assert len(result) == 2
        assert isinstance(result[0].meta.get("symbolic_trace"), dict)
        assert "symbolic_trace" not in result[1].meta

    def test_empty_evidence_returns_empty(self):
        assert attach_symbolic_trace("질문", []) == []


# ---------------------------------------------------------------------------
# retrieve_unconstrained (orchestrator)
# ---------------------------------------------------------------------------
class TestRetrieveUnconstrained:
    def test_merges_and_dedups_sources(self):
        db = FakeNeo4j(chunks=[chunk_record(1), chunk_record(2)])
        route = classify_query("개인정보 처리")
        docs = retrieve_unconstrained(db.run_read, FakeEmbedder(), 4, "개인정보 처리", route)
        assert {d.meta["id"] for d in docs} == {"chunk-1", "chunk-2"}

    def test_rule_kind_does_not_error(self):
        db = FakeNeo4j(chunks=[chunk_record(1)])
        route = classify_query("금지 행위")
        docs = retrieve_unconstrained(db.run_read, FakeEmbedder(), 4, "금지 행위", route)
        assert len(docs) == 1


# ---------------------------------------------------------------------------
# retrieve_vectorrag (orchestrator)
# ---------------------------------------------------------------------------
class TestRetrieveVectorrag:
    def test_merges_exact_vector_and_attaches_context(self):
        context = [{
            "id": "chunk-1",
            "legal_unit_id": "lu-1",
            "legal_unit_kind": "article",
            "legal_unit_number": "제15조",
            "legal_unit_title": "목적",
            "article_id": "a-1",
            "article_number": "제15조",
            "article_title": "목적",
        }]
        db = FakeNeo4j(chunks=[chunk_record(1)], context=context)
        docs = retrieve_vectorrag(db.run_read, FakeEmbedder(), 4, '"개인정보" 보호')
        assert len(docs) == 1
        assert docs[0].meta["document_type"] == "statute"
        assert docs[0].meta["article_number"] == "제15조"


# ---------------------------------------------------------------------------
# retrieve_graphrag (full orchestrator)
# ---------------------------------------------------------------------------
class TestRetrieveGraphrag:
    def test_unconstrained_path(self):
        db = FakeNeo4j(chunks=[chunk_record(1), chunk_record(2)])
        matcher = SourceMatcher(db.run_read)
        docs = retrieve_graphrag(db.run_read, FakeEmbedder(), 4, "개인정보 처리 방침", matcher)
        assert {d.meta["id"] for d in docs} == {"chunk-1", "chunk-2"}

    def test_source_constrained_path(self):
        db = FakeNeo4j(chunks=[chunk_record(1)])
        matcher = FakeSourceMatcher(["documents/raw/statutes/법.pdf"])
        docs = retrieve_graphrag(db.run_read, FakeEmbedder(), 4, "개인정보 보호법", matcher)
        assert len(docs) == 1
        assert docs[0].meta["id"] == "chunk-1"


# ---------------------------------------------------------------------------
# retrieve_chunks (strategy dispatcher)
# ---------------------------------------------------------------------------
class TestRetrieveChunks:
    def test_vector_strategy_returns_chunks(self):
        db = FakeNeo4j(chunks=[chunk_record(1)])
        docs = retrieve_chunks(db.run_read, FakeEmbedder(), 4, "개인정보", FakeSourceMatcher([]), "vector")
        assert len(docs) == 1

    def test_graph_strategy_returns_chunks(self):
        db = FakeNeo4j(chunks=[chunk_record(1)])
        docs = retrieve_chunks(db.run_read, FakeEmbedder(), 4, "개인정보 처리 방침", FakeSourceMatcher([]), "graph")
        assert len(docs) == 1

    def test_neuro_symbolic_attaches_trace(self):
        db = FakeNeo4j(chunks=[chunk_record(1)])
        docs = retrieve_chunks(
            db.run_read, FakeEmbedder(), 4, "개인정보 처리 방침", FakeSourceMatcher([]), "neuro_symbolic"
        )
        assert len(docs) == 1
        assert isinstance(docs[0].meta.get("symbolic_trace"), dict)

    def test_hybrid_alias_resolves_to_graph(self):
        db = FakeNeo4j(chunks=[chunk_record(1)])
        docs = retrieve_chunks(db.run_read, FakeEmbedder(), 4, "개인정보 처리 방침", FakeSourceMatcher([]), "hybrid")
        assert len(docs) == 1

    def test_unknown_strategy_raises(self):
        db = FakeNeo4j(chunks=[chunk_record(1)])
        with pytest.raises(ValueError):
            retrieve_chunks(db.run_read, FakeEmbedder(), 4, "개인정보", FakeSourceMatcher([]), "bogus")
