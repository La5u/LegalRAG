from backend.document import Document
import asyncio

from evaluation import evaluator
from evaluation.evaluator import answer_citation_score, retrieval_quality
from evaluation.schema import EvaluationQuestion


def question() -> EvaluationQuestion:
    return EvaluationQuestion(
        id="test-1",
        source_path="documents/raw/statutes/대상법.hwpx",
        document_type="statute",
        question_type="obligation",
        question="질문",
        gold_answer="정답",
        gold_citations=["제2조"],
    )


def test_answer_citation_requires_answer_to_reference_grounded_chunk():
    chunks = [
        Document(
            "제2조 정답",
            {
                "article_number": "제2조",
                "document_name": "대상법.hwpx",
                "source_path": "documents/raw/statutes/대상법.hwpx",
            },
        )
    ]

    assert answer_citation_score("정답", chunks, question()) == 0.0
    assert answer_citation_score("정답 [1]", chunks, question()) == 1.0


def test_retrieval_quality_separates_document_recall_rank_and_noise():
    chunks = [
        Document("관련 없음", {"source_path": "documents/raw/statutes/다른법.hwpx"}),
        Document(
            "제2조 정답",
            {
                "article_number": "제2조",
                "source_path": "documents/raw/statutes/대상법.hwpx",
            },
        ),
    ]

    document_hit, citation_recall, reciprocal_rank, irrelevant_ratio = retrieval_quality(
        chunks, question()
    )

    assert document_hit == 1.0
    assert citation_recall == 1.0
    assert reciprocal_rank == 0.5
    assert irrelevant_ratio == 0.5


def test_llm_judge_retries_when_hosted_model_returns_no_final_text(monkeypatch):
    budgets = []

    async def fake_stream(request, callback):
        budgets.append(request["options"]["num_predict"])
        if len(budgets) == 2:
            await callback("0.7")

    monkeypatch.setattr(evaluator, "stream_model_payload", fake_stream)

    score = asyncio.run(evaluator.score_llm_judge("질문", "답변", "정답"))

    assert budgets == [512, 2048]
    assert score == 0.7
