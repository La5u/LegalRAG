import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal


EvaluationMode = Literal["exact", "choice", "yes_no", "citation", "llm_judge"]
DocumentType = Literal[
    "statute", "precedent", "regulation", "guidance",
    "contract", "constitutional_decision", "qna",
]
QuestionType = Literal[
    "article_lookup", "definition_lookup", "yes_no", "multiple_choice",
    "condition", "exception", "obligation", "prohibition",
    "citation", "holding", "contract_duty", "regulatory_guidance",
    "comparison", "scenario_apply",
]
Difficulty = Literal["easy", "medium", "hard"]


@dataclass(frozen=True)
class EvaluationQuestion:
    id: str
    source_path: str
    document_type: DocumentType
    question_type: QuestionType
    question: str
    gold_answer: str
    gold_citations: list[str] = field(default_factory=list)
    relevant_terms: list[str] = field(default_factory=list)
    evaluation_mode: EvaluationMode = "exact"
    difficulty: Difficulty = "medium"
    notes: str = ""
    options: list[str] | None = None

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> "EvaluationQuestion":
        data = json.loads(line)
        return cls(**data)


@dataclass(frozen=True)
class EvaluationResult:
    id: str
    strategy: str
    question: str
    gold_answer: str
    gold_citations: list[str]
    evaluation_mode: EvaluationMode
    answer: str
    retrieved_citations: list[str]
    source_precision: float
    score: float | None
    latency: float
    details: str = ""
    symbolic_score: float = 0.0
    retrieved_sources: list[dict] = field(default_factory=list)
    document_hit: float = 0.0
    citation_recall: float = 0.0
    reciprocal_rank: float = 0.0
    irrelevant_ratio: float = 0.0

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> "EvaluationResult":
        data = json.loads(line)
        return cls(**data)


def load_questions(path: str | Path) -> list[EvaluationQuestion]:
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(EvaluationQuestion.from_jsonl(line))
    return questions


def load_results(path: str | Path) -> list[EvaluationResult]:
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(EvaluationResult.from_jsonl(line))
    return results


def append_result(result: EvaluationResult, path: str | Path) -> None:
    results_path = Path(path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(result.to_jsonl() + "\n")
