import argparse
import asyncio
import logging
import re
from pathlib import Path
from time import perf_counter

from neo4j.exceptions import ServiceUnavailable

from evaluation.schema import (
    EvaluationQuestion,
    EvaluationResult,
    append_result,
    load_questions,
    load_results,
)
from backend.config import (
    JUDGE_MODEL,
    RETRIEVAL_STRATEGIES,
    normalize_retrieval_strategy,
)
from backend.document import Document
from backend.llm import stream_payload
from evaluation.metrics import (
    CITATION_PATTERN,
    citation_matches,
    context_precision,
    normalize_metric_text,
    normalize_source_name,
)
from backend.rag import LegalRAG

logger = logging.getLogger(__name__)

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.jsonl"
RESULTS_PATH = Path(__file__).resolve().parent / "results.jsonl"
STRATEGIES = RETRIEVAL_STRATEGIES
VALID_STRATEGIES = set(STRATEGIES)

KOREAN_YES = re.compile(r"^(예|네|그렇다|맞다|응|yes)", re.IGNORECASE)
KOREAN_NO = re.compile(r"^(아니오|아니요|아니다|no|아님)", re.IGNORECASE)

LLM_JUDGE_SYSTEM = (
    "당신은 평가자입니다. 질문에 대해 생성 답변이 정답과 같은 법적 의미인지 비교하세요. "
    "의미 일치도를 0.0부터 1.0까지 점수로 매기세요:\n"
    "- 1.0: 동일하거나 법적 의미가 같음\n"
    "- 0.7: 대체로 맞지만 작은 세부사항이 빠짐\n"
    "- 0.3: 일부만 맞고 중요한 오류나 누락이 있음\n"
    "- 0.0: 틀렸거나 관련 없음\n\n"
    "설명하지 말고 숫자 하나만 출력하세요. 예: 1.0 또는 0.7"
)


def normalize(text: str) -> str:
    return normalize_metric_text(text)


def score_exact(answer: str, gold_answer: str) -> float:
    return 1.0 if normalize(gold_answer) in normalize(answer) else 0.0


def score_yes_no(answer: str, gold_answer: str) -> float:
    cleaned = answer.strip()
    is_yes = bool(KOREAN_YES.match(cleaned))
    is_no = bool(KOREAN_NO.match(cleaned))
    if not is_yes and not is_no:
        return 0.0
    gold_positive = normalize(gold_answer) in ("예", "yes", "네", "그렇다")
    return 1.0 if (is_yes and gold_positive) or (is_no and not gold_positive) else 0.0


def score_citation(
    answer: str, retrieved_citations: list[str], gold_citations: list[str]
) -> float:
    answer_citations = CITATION_PATTERN.findall(answer)
    all_retrieved = retrieved_citations + answer_citations
    found = 0
    for gold_ref in gold_citations:
        if any(citation_matches(gold_ref, citation) for citation in all_retrieved):
            found += 1
    return found / len(gold_citations) if gold_citations else 0.0


async def score_llm_judge(
    question: str, answer: str, gold_answer: str
) -> float:
    prompt = (
        f"질문: {question}\n\n"
        f"정답: {gold_answer}\n\n"
        f"생성 답변: {answer}\n\n"
        f"점수(0.0부터 1.0):"
    )
    request = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": LLM_JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "keep_alive": "1h",
        "options": {
            "temperature": 0,
            "num_predict": 16,
        },
    }
    tokens: list[str] = []
    async def collect(token: str) -> None:
        tokens.append(token)
    try:
        await stream_payload(request, collect)
    except Exception:
        return 0.0
    raw = "".join(tokens).strip()
    try:
        score = float(raw)
        return max(0.0, min(1.0, score))
    except (ValueError, TypeError):
        match = re.search(r"([01](?:\.\d+)?)", raw)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
        return 0.0


def extract_citations(chunks: list[Document]) -> list[str]:
    citations = set()
    for chunk in chunks:
        specific = chunk.meta.get("specific_citation")
        if specific:
            citations.add(specific)
        article = chunk.meta.get("article_number")
        if article:
            citations.add(article)
        source = chunk.meta.get("document_name", "")
        if source:
            citations.add(source)
        content_citations = CITATION_PATTERN.findall(chunk.content)
        for cc in content_citations:
            citations.add(cc)
        for fact in chunk.meta.get("graph_facts", []):
            fact_article = getattr(fact, "article", None)
            if fact_article:
                citations.add(fact_article)
            fact_document = getattr(fact, "document", "")
            if fact_document:
                citations.add(fact_document)
            fact_sentence = getattr(fact, "sentence", "")
            for fact_citation in CITATION_PATTERN.findall(fact_sentence):
                citations.add(fact_citation)
    return sorted(citations)


def source_precision(chunks: list[Document], source_path: str) -> float:
    text_chunks = [
        chunk
        for chunk in chunks
        if not chunk.content.startswith(("Graph fact:", "그래프 사실:"))
    ]
    if not text_chunks:
        return 0.0
    expected = normalize_source_name(Path(source_path).name)
    if not expected:
        return 0.0
    matches = 0
    for chunk in text_chunks:
        names = [
            chunk.meta.get("source_path", ""),
            chunk.meta.get("document_name", ""),
        ]
        if any(
            citation_matches(expected, Path(name).name if name else "")
            for name in names
        ):
            matches += 1
    return matches / len(text_chunks)


def trace_presence(chunks: list[Document]) -> float:
    return 1.0 if any(chunk.meta.get("symbolic_trace") for chunk in chunks) else 0.0


async def evaluate_question(
    rag: LegalRAG,
    question: EvaluationQuestion,
    strategy: str,
    retrieval_only: bool = False,
) -> EvaluationResult:
    started_at = perf_counter()
    chunks = rag.retrieve(question.question, strategy=strategy)
    answer_parts: list[str] = []

    async def collect(token: str) -> None:
        answer_parts.append(token)

    retrieved_citations = extract_citations(chunks)
    source_precision_score = context_precision(chunks, question.gold_citations)
    if not question.gold_citations:
        source_precision_score = source_precision(chunks, question.source_path)
    trace_score = trace_presence(chunks)
    if retrieval_only:
        latency = perf_counter() - started_at
        citation_score = score_citation("", retrieved_citations, question.gold_citations)
        score = citation_score if question.gold_citations else source_precision_score
        return EvaluationResult(
            id=question.id,
            strategy=strategy,
            question=question.question,
            gold_answer=question.gold_answer,
            gold_citations=question.gold_citations,
            evaluation_mode=question.evaluation_mode,
            answer="",
            retrieved_citations=retrieved_citations,
            source_precision=source_precision_score,
            score=score,
            latency=latency,
            details=(
                f"retrieval_only_citation={citation_score:.2f}_"
                f"trace={trace_score:.2f}"
            ),
            symbolic_score=trace_score,
        )

    await rag.generate_async(question.question, chunks, collect)
    latency = perf_counter() - started_at
    answer = "".join(answer_parts)
    score, details = await score_generated_answer(
        question,
        answer,
        retrieved_citations,
    )

    return EvaluationResult(
        id=question.id,
        strategy=strategy,
        question=question.question,
        gold_answer=question.gold_answer,
        gold_citations=question.gold_citations,
        evaluation_mode=question.evaluation_mode,
        answer=answer,
        retrieved_citations=retrieved_citations,
        source_precision=source_precision_score,
        score=score,
        latency=latency,
        details=details,
        symbolic_score=trace_score,
    )


async def score_generated_answer(
    question: EvaluationQuestion,
    answer: str,
    retrieved_citations: list[str],
) -> tuple[float, str]:
    mode = question.evaluation_mode
    if mode == "exact":
        score = score_exact(answer, question.gold_answer)
        return score, "exact_match" if score == 1.0 else "mismatch"
    if mode == "yes_no":
        score = score_yes_no(answer, question.gold_answer)
        return score, "yes_no_correct" if score == 1.0 else "yes_no_incorrect"
    if mode == "citation":
        score = score_citation(answer, retrieved_citations, question.gold_citations)
        return score, f"citation_score_{score:.2f}"
    if mode == "llm_judge":
        citation_ok = (
            score_citation(answer, retrieved_citations, question.gold_citations)
            if question.gold_citations
            else 1.0
        )
        llm_score = await score_llm_judge(
            question.question,
            answer,
            question.gold_answer,
        )
        score = 0.7 * llm_score + 0.3 * citation_ok
        return score, f"llm_judge={llm_score:.2f}_citation={citation_ok:.2f}"
    return 0.0, "unknown_mode"


def print_summary(results: list[EvaluationResult]) -> None:
    if not results:
        return
    strategies = set(r.strategy for r in results)
    for strategy in sorted(strategies):
        strat_results = [r for r in results if r.strategy == strategy]
        exact_scores = [
            r.score for r in strat_results if r.evaluation_mode in ("exact", "yes_no", "citation")
        ]
        llm_scores = [
            r.score for r in strat_results if r.evaluation_mode == "llm_judge"
        ]
        avg_exact = sum(exact_scores) / len(exact_scores) if exact_scores else 0.0
        avg_llm = sum(llm_scores) / len(llm_scores) if llm_scores else 0.0
        avg_precision = sum(r.source_precision for r in strat_results) / len(strat_results)
        avg_trace = sum(r.symbolic_score for r in strat_results) / len(strat_results)
        avg_latency = sum(r.latency for r in strat_results) / len(strat_results)
        print(f"\n[{strategy}]")
        print(f"  Questions: {len(strat_results)}")
        print(f"  Exact/choice/citation avg: {avg_exact:.2f} ({len(exact_scores)} items)")
        print(f"  LLM-judge avg: {avg_llm:.2f} ({len(llm_scores)} items)")
        print(f"  Avg source precision: {avg_precision:.2f}")
        print(f"  Avg trace presence: {avg_trace:.2f}")
        print(f"  Avg latency: {avg_latency:.2f}s")
        low_precision = [r for r in strat_results if r.source_precision < 0.3]
        if low_precision:
            print(f"  Low-precision sources: {len(low_precision)} questions")
            for r in low_precision[:3]:
                print(f"    {r.id}: {r.question[:60]}...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation questions through LegalRAG.")
    parser.add_argument(
        "--questions",
        type=str,
        default=str(QUESTIONS_PATH),
        help="Path to questions JSONL file.",
    )
    parser.add_argument(
        "--results",
        type=str,
        default=str(RESULTS_PATH),
        help="Path to write results JSONL file.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Run only the first N questions.",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=",".join(STRATEGIES),
        help="Comma-separated retrieval strategies to run.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Retrieve and score source coverage without generating answers.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Maximum number of question/strategy evaluations to run at once.",
    )
    return parser.parse_args()


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    questions = load_questions(args.questions)
    strategies = [
        normalize_retrieval_strategy(strategy.strip())
        for strategy in args.strategies.split(",")
        if strategy.strip()
    ]
    unknown_strategies = [
        strategy for strategy in strategies if strategy not in VALID_STRATEGIES
    ]
    if unknown_strategies:
        raise ValueError(f"Unknown strategies: {', '.join(unknown_strategies)}")
    if args.max_questions:
        questions = questions[:args.max_questions]
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    logger.info("Loaded %s questions from %s", len(questions), args.questions)
    results_path = Path(args.results)
    all_results = load_results(results_path) if results_path.exists() else []
    completed = {(result.id, result.strategy) for result in all_results}
    if completed:
        logger.info("Skipping %s completed result rows from %s", len(completed), results_path)
    try:
        with LegalRAG() as rag:
            _ = rag.embedding_model
            semaphore = asyncio.Semaphore(args.concurrency)
            results_lock = asyncio.Lock()

            async def run_one(question: EvaluationQuestion, strategy: str) -> None:
                async with semaphore:
                    logger.info("[%s] %s: %s", strategy, question.id, question.question[:80])
                    result = await evaluate_question(
                        rag,
                        question,
                        strategy,
                        retrieval_only=args.retrieval_only,
                    )
                    async with results_lock:
                        all_results.append(result)
                        append_result(result, args.results)
                        completed.add((question.id, strategy))

            tasks = [
                asyncio.create_task(run_one(question, strategy))
                for question in questions
                for strategy in strategies
                if (question.id, strategy) not in completed
            ]
            if tasks:
                await asyncio.gather(*tasks)
    except ServiceUnavailable as error:
        raise SystemExit(
            "Neo4j is not reachable. Start it with `docker compose up -d neo4j` "
            "or `podman compose up -d neo4j`, then index documents before "
            "running evaluation."
        ) from error
    print_summary(all_results)
    logger.info("Results written to %s", args.results)


if __name__ == "__main__":
    asyncio.run(main())
