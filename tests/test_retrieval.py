import pytest

from backend.config import NEO4J_VECTOR_INDEX
from backend.document import Document
from backend.graph_retrieve import GraphFact
from backend.retrieval import (
    RetrievalUnavailable,
    attach_enumerated_items,
    attach_parent_leads,
    adjacent_query_phrases,
    attach_legal_context,
    expand_legal_units,
    fuse_ranked_results,
    graph_facts_for_route,
    merge_ranked_expansion,
    retrieve_case_source_matches,
    retrieve_contract_matches,
    retrieve_document_summary_matches,
    retrieve_exact_phrase_matches,
    retrieve_legal_unit_matches,
    retrieve_source_constrained_matches,
    retrieve_sparse_chunks,
    select_evidence,
    source_question_focus,
    retrieve_chunks,
    retrieve_unconstrained,
    retrieve_vector_chunks,
    prioritize_symbolic_evidence,
    select_graph_facts,
    specific_legal_citation,
)
from backend.routing import QueryRoute, classify_query


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
        self.last_input = texts
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


class RecordingNeo4j:
    def __init__(self, records=None):
        self.records = records or [chunk_record(1)]
        self.calls = []

    def run_read(self, query, **params):
        self.calls.append({"query": query, "params": params})
        return list(self.records)


class FakeSourceMatcher:
    def __init__(self, source_paths):
        self._paths = list(source_paths)

    def constraints(self, _query):
        return list(self._paths)

    def category_constraints(self, _category):
        return []


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


def test_retrieve_baseline_caps_sources_for_all_modes():
    db = FakeNeo4j(chunks=[chunk_record(index) for index in range(1, 8)])

    result = retrieve_chunks(
        db.run_read,
        FakeEmbedder(),
        6,
        "개인정보 처리",
        FakeSourceMatcher([]),
        "vector",
    )

    assert len(result.evidence) == 4


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

    def test_e5_query_prefix_is_added_before_search(self):
        db = FakeNeo4j(chunks=[chunk_record(1)])
        embedder = FakeEmbedder()
        retrieve_vector_chunks(db.run_read, embedder, top_k=4, query="개인정보")
        assert embedder.last_input == "query: 개인정보"

    def test_raises_when_no_index(self):
        db = FakeNeo4j(chunks=[chunk_record(1)], vector_index_exists=False)
        with pytest.raises(RetrievalUnavailable):
            retrieve_vector_chunks(db.run_read, FakeEmbedder(), top_k=4, query="개인정보")

    def test_source_constraint_overfetches_and_sends_paths(self):
        db = FakeNeo4j(chunks=[chunk_record(1)])

        retrieve_vector_chunks(
            db.run_read,
            FakeEmbedder(),
            top_k=4,
            query="개인정보",
            source_paths=["documents/raw/statutes/법.pdf"],
        )

        vector_call = next(call for call in db.calls if "queryNodes" in call)
        assert "$source_paths = [] OR node.source_path IN $source_paths" in vector_call


# ---------------------------------------------------------------------------
# retrieve_sparse_chunks
# ---------------------------------------------------------------------------
class TestRetrieveSparseChunks:
    def test_no_terms_returns_empty_without_query(self):
        db = FakeNeo4j(chunks=[chunk_record(1)])
        route = QueryRoute(kind="general", search_terms=[])
        assert retrieve_sparse_chunks(db.run_read, route) == []
        assert db.calls == []


# ---------------------------------------------------------------------------
# retrieve_legal_unit_matches
# ---------------------------------------------------------------------------
class TestRetrieveLegalUnitMatches:
    def test_no_terms_or_articles_returns_empty_without_query(self):
        db = FakeNeo4j(chunks=[chunk_record(1)])
        route = QueryRoute(kind="general", search_terms=[], article_numbers=[])
        assert retrieve_legal_unit_matches(db.run_read, route) == []
        assert db.calls == []


# ---------------------------------------------------------------------------
# retrieve_source_constrained_matches / summary / contract / case
# ---------------------------------------------------------------------------
class TestRetrieverQueries:
    def test_sparse_chunks_send_normalized_terms_and_limit(self):
        db = RecordingNeo4j()
        route = QueryRoute(kind="general", search_terms=["개인정보", "처리"])

        docs = retrieve_sparse_chunks(db.run_read, route)

        assert [doc.meta["id"] for doc in docs] == ["chunk-1"]
        assert db.calls[0]["params"]["terms"] == ["개인정보", "처리"]
        assert db.calls[0]["params"]["limit"] == 8

    def test_legal_unit_matches_send_articles_terms_and_source_scope(self):
        db = RecordingNeo4j()
        route = QueryRoute(kind="article_lookup", search_terms=["개인정보"], article_numbers=["제15조"])

        docs = retrieve_legal_unit_matches(db.run_read, route, ["documents/raw/statutes/법.pdf"])

        params = db.calls[0]["params"]
        assert [doc.meta["id"] for doc in docs] == ["chunk-1"]
        assert params["terms"] == ["개인정보"]
        assert params["article_numbers"] == ["제15조"]
        assert params["source_paths"] == ["documents/raw/statutes/법.pdf"]
        assert params["article_limit"] == 8

    def test_source_specific_retrievers_send_distinct_scoring_inputs(self):
        db = RecordingNeo4j()
        source_paths = ["documents/raw/contracts/계약서.pdf"]
        route = QueryRoute(kind="general", search_terms=["임대인"])

        retrieve_source_constrained_matches(db.run_read, "첫 부분 임대인 의무", route, source_paths)
        retrieve_document_summary_matches(db.run_read, "주요 내용", source_paths)
        retrieve_contract_matches(db.run_read, "임대인 의무", source_paths)
        retrieve_case_source_matches(db.run_read, "판례 판단 이유", source_paths)

        source_params = db.calls[0]["params"]
        summary_params = db.calls[1]["params"]
        contract_params = db.calls[2]["params"]
        case_params = db.calls[3]["params"]
        assert source_params["source_paths"] == source_paths
        assert source_params["prefer_start"] is True
        assert "임대인" in source_params["terms"]
        assert summary_params["source_paths"] == source_paths
        assert "주요" in summary_params["terms"]
        assert {"임대인", "임차인"}.issubset(contract_params["party_terms"])
        assert "판단" in case_params["case_terms"]
        assert "이유" in case_params["case_terms"]

    def test_source_matches_send_surface_word_phrases(self):
        db = RecordingNeo4j()
        route = QueryRoute(kind="prohibition", search_terms=["부정한", "목적"])

        retrieve_source_constrained_matches(
            db.run_read,
            "부정한 목적으로 상호를 사용할 수 있나요?",
            route,
            ["documents/raw/statutes/상법.hwpx"],
        )

        assert "부정한 목적으로" in db.calls[0]["params"]["phrases"]


def test_source_scoped_retrieval_uses_question_after_document_prefix():
    assert source_question_focus(
        "긴 법률 시행령에 따르면, 본인확인기관의 기술적 능력 요건은?",
    ) == "본인확인기관의 기술적 능력 요건은?"


def test_adjacent_query_phrases_preserve_conjugated_surface_forms():
    assert "부정한 목적으로" in adjacent_query_phrases(
        "부정한 목적으로 상호를 사용할 수 있나요?"
    )

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


class TestAttachEnumeratedItems:
    def test_appends_structural_children_to_parent(self):
        parent = Document(
            "다음 각 호의 사업에는 적용하지 않는다.",
            {"id": "parent", "chunk_number": 1},
        )
        children = [
            {**chunk_record(2, content="1. 여객운송업"), "seed_id": "parent", "rank": 1},
            {**chunk_record(3, content="2. 전기사업"), "seed_id": "parent", "rank": 2},
        ]

        result = attach_enumerated_items(lambda _query, **_params: children, [parent])

        assert "1. 여객운송업" in result[0].content
        assert "2. 전기사업" in result[0].content
        assert result[0].meta["enumerated_item_count"] == 2

    def test_skips_query_when_no_list_lead_in(self):
        calls = []
        document = Document("일반 조문", {"id": "ordinary"})

        result = attach_enumerated_items(lambda *args, **kwargs: calls.append(args), [document])

        assert result == [document]
        assert calls == []


class TestAttachParentLeads:
    def test_prepends_governing_text_to_numbered_child(self):
        child = Document(
            "1. 재화를 반환받은 날",
            {
                "id": "child",
                "legal_unit_kind": "item",
                "chunk_number": 2,
            },
        )
        records = [{
            "child_id": "child",
            "lead_id": "parent",
            "lead_content": "대금은 3영업일 이내에 환급하여야 한다.",
        }]

        result = attach_parent_leads(lambda _query, **_params: records, [child])

        assert result[0].content.startswith("대금은 3영업일 이내")
        assert result[0].content.endswith("1. 재화를 반환받은 날")
        assert result[0].meta["parent_lead_id"] == "parent"

    def test_does_not_query_for_non_child_chunks(self):
        calls = []
        article = Document(
            "제18조 본문",
            {"id": "article", "legal_unit_kind": "article"},
        )

        result = attach_parent_leads(
            lambda *args, **kwargs: calls.append((args, kwargs)),
            [article],
        )

        assert result == [article]
        assert calls == []


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


class TestSelectEvidence:
    def test_applies_shared_chunk_and_word_budget(self):
        documents = [
            Document("one two", {"id": "1"}),
            Document("three four", {"id": "2"}),
            Document("five six", {"id": "3"}),
        ]

        selected = select_evidence(documents, top_k=3, max_words=4)

        assert [document.meta["id"] for document in selected] == ["1", "2"]


class TestFuseRankedResults:
    def test_vector_outweighs_separate_term_only_matches(self):
        vector = Document("semantic", {"id": "vector"}, 0.8)
        legal_term = Document("legal term", {"id": "legal"}, 5.0)
        sparse_term = Document("sparse term", {"id": "sparse"}, 8.0)

        fused = fuse_ranked_results([
            ("vector", [vector]),
            ("legal_unit", [legal_term]),
            ("sparse_terms", [sparse_term]),
        ])

        assert fused[0].meta["id"] == "vector"
        assert fused[0].meta["retrieval_channels"] == ["vector"]

    def test_agreement_across_channels_increases_rank(self):
        shared = Document("shared", {"id": "shared"})
        vector_only = Document("vector", {"id": "vector"})

        fused = fuse_ranked_results([
            ("vector", [vector_only, shared]),
            ("legal_unit", [shared]),
        ])

        assert fused[0].meta["id"] == "shared"
        assert fused[0].meta["retrieval_channels"] == ["vector", "legal_unit"]


# ---------------------------------------------------------------------------
# select_graph_facts (pure)
# ---------------------------------------------------------------------------
class TestSelectGraphFacts:
    def test_returns_sorted_facts(self):
        facts = [_graph_fact(score=0.5, path="P1"), _graph_fact(score=0.9, path="P2")]
        result = select_graph_facts(facts, top_k=2)
        assert len(result) == 2
        assert [fact.score for fact in result] == [0.9, 0.5]

    def test_dedups_by_key(self):
        facts = [_graph_fact(score=0.5, path="P"), _graph_fact(score=0.9, path="P")]
        result = select_graph_facts(facts, top_k=2)
        assert len(result) == 1
        assert result[0].score == 0.9

    def test_no_facts_returns_empty(self):
        assert select_graph_facts([], top_k=2) == []


def test_graph_facts_for_route_keeps_only_answer_capable_relations():
    duty = _graph_fact(path="duty")
    prohibition = GraphFact(
        id="prohibition",
        document=duty.document,
        source_path=duty.source_path,
        chunk=duty.chunk,
        relation="PROHIBITS",
        path="prohibition",
        sentence=duty.sentence,
        score=duty.score,
    )

    result = graph_facts_for_route(
        [duty, prohibition], QueryRoute(kind="obligation")
    )

    assert result == [duty]


def test_graph_facts_for_route_omits_relations_for_direct_article_lookup():
    assert graph_facts_for_route(
        [_graph_fact()], QueryRoute(kind="article_lookup")
    ) == []


def test_graph_facts_for_route_omits_unverified_exception_edges():
    fact = GraphFact(
        id="exception",
        document="law.hwpx",
        source_path="law.hwpx",
        chunk=1,
        relation="EXCEPTION_TO",
        path="exception",
        sentence="다만 예외로 한다.",
        score=0.9,
    )

    assert graph_facts_for_route([fact], QueryRoute(kind="exception")) == []


# ---------------------------------------------------------------------------
# prioritize_symbolic_evidence
# ---------------------------------------------------------------------------
class TestPrioritizeSymbolicEvidence:
    def test_returns_trace_separately_from_evidence(self):
        evidence = [
            Document("개인정보를 보호한다", {"id": "1"}),
            Document("기타 내용", {"id": "2"}),
        ]
        ordered, trace = prioritize_symbolic_evidence(
            "개인정보 보호", evidence, [],
        )
        assert len(ordered) == 2
        assert isinstance(trace, dict)
        assert all("symbolic_trace" not in chunk.meta for chunk in ordered)

    def test_empty_evidence_returns_empty(self):
        assert prioritize_symbolic_evidence("질문", [], []) == ([], None)


# ---------------------------------------------------------------------------
# retrieve_unconstrained (orchestrator)
# ---------------------------------------------------------------------------
class TestRetrieveUnconstrained:
    def test_merges_and_dedups_sources(self):
        db = FakeNeo4j(chunks=[chunk_record(1), chunk_record(2)])
        route = classify_query("개인정보 처리")
        docs = retrieve_unconstrained(db.run_read, FakeEmbedder(), 4, "개인정보 처리", route)
        assert {d.meta["id"] for d in docs} == {"chunk-1", "chunk-2"}


# ---------------------------------------------------------------------------
# retrieve_chunks (strategy dispatcher)
# ---------------------------------------------------------------------------
class TestRetrieveChunks:
    def test_vector_mode_does_not_run_legal_or_graph_retrievers(self):
        db = FakeNeo4j(chunks=[chunk_record(1)])

        retrieve_chunks(
            db.run_read, FakeEmbedder(), 4, "개인정보 의무", FakeSourceMatcher([]), "vector"
        )

        assert not any("lexical_score" in query for query in db.calls)
        assert not any("MATCH (source:Concept)" in query for query in db.calls)

    def test_neuro_symbolic_attaches_trace(self):
        db = FakeNeo4j(chunks=[chunk_record(1)])
        result = retrieve_chunks(
            db.run_read, FakeEmbedder(), 4, "개인정보 처리 방침", FakeSourceMatcher([]), "neuro_symbolic"
        )
        assert len(result.evidence) == 1
        assert isinstance(result.symbolic_trace, dict)
        assert all("symbolic_trace" not in chunk.meta for chunk in result.evidence)

    def test_unknown_strategy_raises(self):
        db = FakeNeo4j(chunks=[chunk_record(1)])
        with pytest.raises(ValueError):
            retrieve_chunks(db.run_read, FakeEmbedder(), 4, "개인정보", FakeSourceMatcher([]), "bogus")

    def test_conflicting_named_source_and_category_returns_no_evidence(self):
        class ConflictingSourceMatcher:
            def constraints(self, _query):
                return ["documents/raw/statutes/법.pdf"]

            def category_constraints(self, _category):
                return ["documents/raw/contracts/계약서.pdf"]

        db = FakeNeo4j(chunks=[chunk_record(1)])

        result = retrieve_chunks(
            db.run_read,
            FakeEmbedder(),
            4,
            "법에서 무엇을 정하나요?",
            ConflictingSourceMatcher(),
            "vector",
            "contract",
        )

        assert result.evidence == []
        assert db.calls == []
