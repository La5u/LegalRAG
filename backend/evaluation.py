import argparse
import asyncio
from dataclasses import dataclass
from time import perf_counter

from backend.ingest import Document
from backend.rag import LegalRAG


OUTPUT_PATH = "evaluation_results.md"
STRATEGIES = ("vector", "hybrid")


@dataclass(frozen=True)
class EvaluationQuestion:
    document: str
    kind: str
    question: str
    relevant_terms: list[str]


QUESTIONS = [
    EvaluationQuestion(
        "개인정보 보호법 시행령",
        "lookup",
        "개인정보 보호법 시행령에서 영상정보처리기기 안내판에는 어떤 사항을 적어야 하나?",
        ["영상정보처리기기", "안내판", "설치 목적", "촬영 범위"],
    ),
    EvaluationQuestion(
        "개인정보 보호법 시행령",
        "relationship",
        "개인정보 보호법 시행령에서 개인정보처리자가 정보주체에게 알려야 하는 사항은 무엇인가?",
        ["개인정보처리자", "정보주체", "알려야", "통지"],
    ),
    EvaluationQuestion(
        "개인정보 보호법",
        "lookup",
        "개인정보 보호법에서 개인정보의 수집ㆍ이용 동의와 관련된 기본 요건은 무엇인가?",
        ["개인정보", "수집", "이용", "동의"],
    ),
    EvaluationQuestion(
        "개인정보 보호법",
        "relationship",
        "개인정보 보호법에서 개인정보처리자는 정보주체에게 무엇을 통지하거나 알려야 하나?",
        ["개인정보처리자", "정보주체", "통지", "알려야"],
    ),
    EvaluationQuestion(
        "대한민국헌법",
        "lookup",
        "대한민국헌법에서 국민의 자유와 권리는 어떤 경우에 법률로 제한할 수 있나?",
        ["자유", "권리", "법률", "제한"],
    ),
    EvaluationQuestion(
        "대한민국헌법",
        "relationship",
        "대한민국헌법에서 국회와 정부는 법률안과 관련해 어떤 관계를 가지나?",
        ["국회", "정부", "법률안", "제출"],
    ),
    EvaluationQuestion(
        "민법",
        "lookup",
        "민법에서 권리능력은 언제부터 인정되는가?",
        ["권리능력", "생존", "동안"],
    ),
    EvaluationQuestion(
        "민법",
        "relationship",
        "민법에서 법정대리인과 제한능력자의 법률행위는 어떤 관계인가?",
        ["법정대리인", "제한능력자", "동의", "취소"],
    ),
    EvaluationQuestion(
        "민사소송법",
        "lookup",
        "민사소송법에서 관할은 어떤 기준으로 정해지는가?",
        ["관할", "법원", "소"],
    ),
    EvaluationQuestion(
        "민사소송법",
        "relationship",
        "민사소송법에서 당사자와 법원은 소송절차에서 어떤 의무나 권한을 가지나?",
        ["당사자", "법원", "소송절차", "권한"],
    ),
    EvaluationQuestion(
        "인공지능 발전과 신뢰 기반 조성 등에 관한 기본법",
        "lookup",
        "인공지능 발전과 신뢰 기반 조성 등에 관한 기본법에서 고영향 인공지능은 무엇인가?",
        ["고영향", "인공지능", "정의"],
    ),
    EvaluationQuestion(
        "인공지능 발전과 신뢰 기반 조성 등에 관한 기본법",
        "relationship",
        "인공지능 발전과 신뢰 기반 조성 등에 관한 기본법에서 사업자는 이용자에게 무엇을 고지해야 하나?",
        ["사업자", "이용자", "고지", "인공지능"],
    ),
    EvaluationQuestion(
        "정보통신망 이용촉진 및 정보보호 등에 관한 법률",
        "lookup",
        "정보통신망법에서 정보통신서비스 제공자는 개인정보 보호를 위해 무엇을 해야 하나?",
        ["정보통신서비스 제공자", "개인정보", "보호"],
    ),
    EvaluationQuestion(
        "정보통신망 이용촉진 및 정보보호 등에 관한 법률",
        "relationship",
        "정보통신망법에서 이용자와 정보통신서비스 제공자는 개인정보 처리에서 어떤 관계인가?",
        ["이용자", "정보통신서비스 제공자", "개인정보", "동의"],
    ),
    EvaluationQuestion(
        "지능정보화 기본법",
        "lookup",
        "지능정보화 기본법에서 지능정보사회란 무엇인가?",
        ["지능정보사회", "지능정보기술", "정의"],
    ),
    EvaluationQuestion(
        "지능정보화 기본법",
        "relationship",
        "지능정보화 기본법에서 국가와 지방자치단체는 지능정보사회 구현을 위해 어떤 책무를 가지나?",
        ["국가", "지방자치단체", "책무", "지능정보사회"],
    ),
    EvaluationQuestion(
        "행정기본법",
        "lookup",
        "행정기본법에서 행정의 법 원칙은 무엇인가?",
        ["행정", "법 원칙", "비례", "평등"],
    ),
    EvaluationQuestion(
        "행정기본법",
        "relationship",
        "행정기본법에서 행정청은 처분과 관련하여 국민에게 어떤 의무를 지니나?",
        ["행정청", "처분", "국민", "의무"],
    ),
    EvaluationQuestion(
        "행정절차법",
        "lookup",
        "행정절차법에서 처분의 사전통지는 어떤 내용을 포함해야 하나?",
        ["처분", "사전통지", "내용"],
    ),
    EvaluationQuestion(
        "행정절차법",
        "relationship",
        "행정절차법에서 행정청은 당사자에게 의견제출 기회를 어떻게 주어야 하나?",
        ["행정청", "당사자", "의견제출", "기회"],
    ),
    EvaluationQuestion(
        "형법",
        "lookup",
        "형법에서 죄형법정주의와 관련된 기본 원칙은 무엇인가?",
        ["죄", "형벌", "법률"],
    ),
    EvaluationQuestion(
        "형법",
        "relationship",
        "형법에서 행위자와 피해자 또는 법익 침해 사이의 관계를 설명하는 조항은 무엇인가?",
        ["행위자", "피해자", "법익", "침해"],
    ),
    EvaluationQuestion(
        "형사소송법",
        "lookup",
        "형사소송법에서 관할이전의 신청은 어떤 경우에 할 수 있나?",
        ["관할이전", "신청", "직근 상급법원"],
    ),
    EvaluationQuestion(
        "형사소송법",
        "relationship",
        "형사소송법에서 검사와 피고인은 관할이전 신청에서 어떤 관계인가?",
        ["검사", "피고인", "관할이전", "신청"],
    ),
    EvaluationQuestion(
        "US Constitution",
        "lookup",
        "Which article mentions the president's minimum age?",
        ["President", "Age", "thirty five Years", "ARTICLE 2"],
    ),
    EvaluationQuestion(
        "US Constitution",
        "lookup",
        "What is in Article 5 of the US Constitution?",
        ["ARTICLE FIVE", "Amendments", "ratified", "three fourths"],
    ),
    EvaluationQuestion(
        "US Constitution",
        "definition",
        "What is treason against the United States defined as?",
        ["Treason", "shall consist only", "levying War", "Aid and Comfort"],
    ),
    EvaluationQuestion(
        "US Constitution",
        "relationship",
        "What power does Congress have over amendments in Article 5?",
        ["Congress", "propose Amendments", "two thirds", "Convention"],
    ),
]


def context_precision(retrieved_chunks: list[Document], relevant_terms: list[str]) -> float:
    """Return the share of retrieved chunks that contain at least one expected term."""
    text_chunks = [
        chunk
        for chunk in retrieved_chunks
        if not chunk.content.startswith("Graph fact:")
    ]
    if not text_chunks:
        return 0.0

    normalized_terms = [
        term.casefold()
        for term in relevant_terms
        if term.strip()
    ]
    if not normalized_terms:
        return 0.0

    relevant_count = sum(
        1
        for chunk in text_chunks
        if any(term in chunk.content.casefold() for term in normalized_terms)
    )
    return relevant_count / len(text_chunks)


def compact(text: str, limit: int = 180) -> str:
    text = " ".join(text.split())
    text = text.replace("|", "\\|")
    if len(text) <= limit:
        return text
    return f"{text[:limit - 3]}..."


def source_summary(chunks: list[Document]) -> str:
    parts = [
        f"{chunk.meta['document_name']}#{chunk.meta['chunk_number']}"
        for chunk in chunks[:3]
    ]
    return "<br>".join(parts)


async def answer_question(
    rag: LegalRAG,
    evaluation_question: EvaluationQuestion,
    strategy: str,
    retrieval_only: bool,
) -> dict:
    started_at = perf_counter()
    chunks = rag.retrieve(evaluation_question.question, strategy=strategy)
    answer_parts = []

    async def collect(token: str) -> None:
        answer_parts.append(token)

    if not retrieval_only:
        await rag.generate_async(evaluation_question.question, chunks, collect)
    latency = perf_counter() - started_at
    graph_facts = sum(
        1
        for chunk in chunks
        if chunk.content.startswith("Graph fact:")
    )
    return {
        "document": evaluation_question.document,
        "kind": evaluation_question.kind,
        "strategy": strategy,
        "question": evaluation_question.question,
        "source_precision": context_precision(chunks, evaluation_question.relevant_terms),
        "latency": latency,
        "sources": source_summary(chunks),
        "graph_facts": graph_facts,
        "complexity": "Low" if strategy == "vector" else "High",
        "answer": "".join(answer_parts),
    }


def markdown_table(rows: list[dict], retrieval_only: bool) -> str:
    lines = [
        "# Evaluation Results",
        "",
        "Manual score columns use 0-2: 0 = poor, 1 = partial, 2 = good.",
        f"Mode: {'retrieval only' if retrieval_only else 'retrieval plus answer generation'}",
        "",
        "| Document | Type | Strategy | Question | Text source precision | Latency | Graph facts | Complexity | Answer quality | Multi-hop | Explainability | Sources | Answer |",
        "|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['document']} | "
            f"{row['kind']} | "
            f"{row['strategy']} | "
            f"{compact(row['question'], 120)} | "
            f"{row['source_precision']:.2f} | "
            f"{row['latency']:.2f}s | "
            f"{row['graph_facts']} | "
            f"{row['complexity']} | "
            " | "
            " | "
            " | "
            f"{row['sources']} | "
            f"{compact(row['answer'])} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Measure retrieval without waiting for Ollama answer generation.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Run only the first N questions.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    rag = LegalRAG()
    _ = rag.embedding_model
    rows = []
    questions = QUESTIONS[:args.max_questions] if args.max_questions else QUESTIONS
    for question in questions:
        for strategy in STRATEGIES:
            print(f"{strategy}: {question.question}")
            rows.append(
                await answer_question(
                    rag,
                    question,
                    strategy,
                    retrieval_only=args.retrieval_only,
                )
            )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        file.write(markdown_table(rows, args.retrieval_only))
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
