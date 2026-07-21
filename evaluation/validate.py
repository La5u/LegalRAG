import re
import sys
from pathlib import Path
from typing import get_args

from evaluation.schema import (
    Difficulty,
    DocumentType,
    EvaluationMode,
    QuestionType,
    load_questions,
)

VALID_EVALUATION_MODES: set[EvaluationMode] = set(get_args(EvaluationMode))
VALID_DOCUMENT_TYPES = set(get_args(DocumentType))
VALID_QUESTION_TYPES = set(get_args(QuestionType))
VALID_DIFFICULTIES = set(get_args(Difficulty))
VALID_ID_PREFIXES = ("eval-", "codex-")
DOT_PATTERN = re.compile(r"\.\s*$")
HARD_TRUNCATION_LENGTHS = {300, 360, 400, 419, 420}


def validate(path: str | None = None) -> int:
    questions_path = Path(path) if path else Path(__file__).resolve().parent / "questions.jsonl"
    candidate_mode = "candidate" in questions_path.stem
    if not questions_path.exists():
        print(f"MISSING: {questions_path}")
        return 1
    questions = load_questions(str(questions_path))
    if not questions:
        print("ERROR: No questions loaded from questions.jsonl")
        return 1
    errors = 0
    warnings = 0
    seen_ids = set()
    project_root = Path(__file__).resolve().parent.parent
    for i, q in enumerate(questions):
        if q.id in seen_ids:
            print(f"  DUPLICATE ID [{i}]: {q.id}")
            errors += 1
        seen_ids.add(q.id)
        if not q.id.startswith(VALID_ID_PREFIXES):
            print(f"  BAD ID [{i}]: {q.id} (should start with a known prefix)")
            errors += 1
        if q.evaluation_mode not in VALID_EVALUATION_MODES:
            print(f"  BAD evaluation_mode [{q.id}]: {q.evaluation_mode}")
            errors += 1
        if q.document_type not in VALID_DOCUMENT_TYPES:
            print(f"  BAD document_type [{q.id}]: {q.document_type}")
            errors += 1
        if q.question_type not in VALID_QUESTION_TYPES:
            print(f"  BAD question_type [{q.id}]: {q.question_type}")
            errors += 1
        if q.difficulty not in VALID_DIFFICULTIES:
            print(f"  BAD difficulty [{q.id}]: {q.difficulty}")
            errors += 1
        if not q.question.strip():
            print(f"  EMPTY question [{q.id}]")
            errors += 1
        if not (project_root / q.source_path).exists():
            print(f"  MISSING source [{q.id}]: {q.source_path}")
            errors += 1
        if not q.gold_citations:
            print(f"  MISSING gold citation [{q.id}]")
            if candidate_mode:
                warnings += 1
            else:
                errors += 1
        if len(q.gold_answer) in HARD_TRUNCATION_LENGTHS:
            print(f"  TRUNCATED gold answer [{q.id}]")
            if candidate_mode:
                warnings += 1
            else:
                errors += 1
        if not q.gold_answer.strip() and q.evaluation_mode != "llm_judge":
            print(f"  EMPTY gold_answer for non-llm_judge [{q.id}]")
            warnings += 1
        if q.evaluation_mode == "exact" and len(q.gold_answer) < 2:
            print(f"  SHORT gold_answer for exact mode [{q.id}]: {q.gold_answer!r}")
            warnings += 1
        if q.question != q.question.strip():
            print(f"  LEADING/TRAILING WHITESPACE in question [{q.id}]")
            warnings += 1
        if q.evaluation_mode == "choice" and not q.options:
            print(f"  NO options for choice mode [{q.id}]")
            errors += 1
        if DOT_PATTERN.search(q.question.rstrip()):
            print(f"  QUESTION ENDS WITH DOT [{q.id}]: questions should not end with period")
            warnings += 1
    if errors:
        print(f"\nFAILED: {errors} errors, {warnings} warnings across {len(questions)} questions")
        return 1
    print(f"PASS: {len(questions)} questions validated ({warnings} warnings)")
    return 0


if __name__ == "__main__":
    sys.exit(validate(sys.argv[1] if len(sys.argv) > 1 else None))
