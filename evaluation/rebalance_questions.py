import json
import re
from dataclasses import asdict
from pathlib import Path

from evaluation.schema import EvaluationQuestion, load_questions


ROOT = Path(__file__).resolve().parent
CURATED_PATH = ROOT / "questions.jsonl"
CODEX_PATH = ROOT / "codex_candidates.jsonl"
BACKUP_PATH = ROOT / "questions.before_rebalance.jsonl"

REMOVED_SIMILAR_SOURCES = {
    "documents/raw/regulations/개인정보의 안전성 확보조치 기준(개인정보보호위원회고시)(제2025-9호)(20251031).pdf",
    "documents/raw/regulations/소비자분쟁해결기준(공정거래위원회고시)(제2025-14호)(20251218).pdf",
    "documents/raw/regulations/신용정보업감독규정(금융위원회고시)(제2025-2호)(20250205).pdf",
    "documents/raw/regulations/전자금융감독규정(금융위원회고시)(제2026-7호)(20260213).pdf",
    "documents/raw/regulations/표준 개인정보 보호지침(개인정보보호위원회고시)(제2025-4호)(20250411).pdf",
}
GENERIC_PATTERNS = (
    "관련된 핵심 내용을 설명",
    "의무를 부과하는 규정 또는 문장을 하나 설명",
    "어떤 계약 관계나 의무를 정리",
    "판단 또는 주문에서 확인되는 결론",
    "lbox와 관련된 핵심",
    "첫 부분에서 확인되는 적용 대상이나 목적",
)
MALFORMED_PATTERNS = (
    re.compile(r"의 의무는 무엇인가요"),
    re.compile(r"에 따라\s+각 호\s+에 따라"),
)
PER_SOURCE_LIMIT = 6
TARGET_TOTAL = 360


ADDED_QUESTIONS = [
    {
        "question_type": "citation",
        "document_type": "regulation",
        "source_path": "documents/raw/regulations/표준 개인정보 보호지침(고시)(제2025-4호)(20250411).hwpx",
        "question": "표준 개인정보 보호지침 제1조는 어떤 개인정보 보호법 조항에 따른 개인정보 처리 기준을 정한다고 하나요?",
        "gold_answer": "개인정보 보호법 제12조제1항",
        "gold_citations": ["제1조", "제12조제1항"],
        "evaluation_mode": "llm_judge",
        "difficulty": "hard",
        "notes": "Citation question; GraphRAG should help by following the cited statute article.",
    },
    {
        "question_type": "scenario_apply",
        "document_type": "statute",
        "source_path": "documents/raw/statutes/개인정보 보호법(법률)(제20897호)(20251002).hwpx",
        "question": "개인정보처리자가 계약 이행에 꼭 필요한 개인정보를 정보주체 동의 없이 수집할 수 있나요?",
        "gold_answer": "예. 정보주체와 체결한 계약을 이행하거나 계약 체결 과정에서 정보주체 요청에 따른 조치를 이행하기 위하여 필요한 경우에는 동의 없이 개인정보를 수집ㆍ이용할 수 있다.",
        "gold_citations": ["제15조제1항제4호"],
        "evaluation_mode": "llm_judge",
        "difficulty": "hard",
        "notes": "Conditional legal question; Neuro-Symbolic should classify yes with a condition.",
    },
    {
        "question_type": "scenario_apply",
        "document_type": "statute",
        "source_path": "documents/raw/statutes/개인정보 보호법(법률)(제20897호)(20251002).hwpx",
        "question": "가명정보를 처리하다가 특정 개인을 알아보기 위한 목적으로 사용해도 되나요?",
        "gold_answer": "아니요. 개인정보처리자는 가명정보를 처리하는 과정에서 특정 개인을 알아보기 위한 목적으로 가명정보를 처리해서는 안 된다.",
        "gold_citations": ["제28조의5제1항"],
        "evaluation_mode": "llm_judge",
        "difficulty": "hard",
        "notes": "Neuro-symbolic-best prohibition question requiring rule application.",
    },
    {
        "question_type": "scenario_apply",
        "document_type": "statute",
        "source_path": "documents/raw/statutes/민법(법률)(제21454호)(20260317).hwpx",
        "question": "미성년자가 법정대리인 동의 없이 권리만 얻는 계약을 했다면 그 행위는 취소할 수 있나요?",
        "gold_answer": "아니요. 미성년자의 법률행위에는 원칙적으로 법정대리인의 동의가 필요하지만, 권리만을 얻거나 의무만을 면하는 행위는 예외적으로 동의 없이도 유효하다.",
        "gold_citations": ["제5조제1항"],
        "evaluation_mode": "llm_judge",
        "difficulty": "hard",
        "notes": "Exception question where Neuro-Symbolic should apply the proviso.",
    },
    {
        "question_type": "condition",
        "document_type": "statute",
        "source_path": "documents/raw/statutes/형법(법률)(제21450호)(20260312).hwpx",
        "question": "심신장애 때문에 사물을 변별할 능력이 없으면 형법상 처벌되나요?",
        "gold_answer": "아니요. 심신장애로 사물을 변별할 능력이 없거나 의사를 결정할 능력이 없는 자의 행위는 벌하지 않는다.",
        "gold_citations": ["제10조제1항"],
        "evaluation_mode": "llm_judge",
        "difficulty": "medium",
        "notes": "Conditional yes/no-style legal question.",
    },
    {
        "question_type": "comparison",
        "document_type": "statute",
        "source_path": "documents/raw/statutes/약관의 규제에 관한 법률(법률)(제20239호)(20240807).hwpx",
        "question": "약관법에서 약관 작성 의무와 약관 설명 의무는 각각 어느 조항에 근거하나요?",
        "gold_answer": "약관 작성 의무는 제3조제1항에, 약관 설명 의무는 제3조제3항에 근거한다.",
        "gold_citations": ["제3조제1항", "제3조제3항"],
        "evaluation_mode": "llm_judge",
        "difficulty": "hard",
        "notes": "Multi-hop comparison across two clauses in the same article.",
    },
    {
        "question_type": "scenario_apply",
        "document_type": "statute",
        "source_path": "documents/raw/statutes/표시ㆍ광고의 공정화에 관한 법률(법률)(제20712호)(20250121).hwpx",
        "question": "사업자가 경쟁 상품보다 자기 상품이 우수하다고 근거 없이 비교 광고하면 표시광고법상 문제가 되나요?",
        "gold_answer": "예. 부당하게 비교하는 표시ㆍ광고는 부당한 표시ㆍ광고 행위의 한 유형으로 금지된다.",
        "gold_citations": ["제3조제1항"],
        "evaluation_mode": "llm_judge",
        "difficulty": "medium",
        "notes": "Vector-fail candidate because the question paraphrases the statutory label.",
    },
    {
        "question_type": "scenario_apply",
        "document_type": "qna",
        "source_path": "documents/raw/qna/해고근로자_생활법령_QA.pdf",
        "question": "사용자가 근로자를 해고하면서 30일 전에 예고하지 않았다면 어떤 문제가 생기나요?",
        "gold_answer": "사용자는 원칙적으로 적어도 30일 전에 해고예고를 해야 하며, 이를 하지 않으면 해고예고수당 문제가 발생할 수 있다.",
        "gold_citations": ["해고근로자_생활법령_QA.pdf"],
        "evaluation_mode": "llm_judge",
        "difficulty": "medium",
        "notes": "GraphRAG should help if source-constrained guidance retrieval works.",
    },
    {
        "question_type": "obligation",
        "document_type": "statute",
        "source_path": "documents/raw/statutes/개인정보 보호법(법률)(제20897호)(20251002).hwpx",
        "question": "개인정보처리자는 개인정보 처리방침을 수립하거나 변경할 때 어떻게 해야 하나요?",
        "gold_answer": "개인정보처리자는 개인정보 처리방침을 수립하거나 변경하는 경우 정보주체가 쉽게 확인할 수 있도록 대통령령으로 정하는 방법에 따라 공개하여야 한다.",
        "gold_citations": ["제30조제2항"],
        "evaluation_mode": "llm_judge",
        "difficulty": "medium",
        "notes": "Obligation question with a clear duty and source article.",
    },
    {
        "question_type": "obligation",
        "document_type": "regulation",
        "source_path": "documents/raw/regulations/개인정보의 안전성 확보조치 기준(고시)(제2025-9호)(20251031).hwpx",
        "question": "개인정보처리자는 개인정보처리시스템의 접속기록을 관리하기 위해 어떤 조치를 해야 하나요?",
        "gold_answer": "개인정보처리자는 개인정보처리시스템 접속기록을 기록ㆍ보관하고, 접속기록을 점검하는 등 개인정보의 안전성 확보에 필요한 조치를 해야 한다.",
        "gold_citations": ["제8조"],
        "evaluation_mode": "llm_judge",
        "difficulty": "hard",
        "notes": "Obligation question from a regulation, useful for graph relation checking.",
    },
    {
        "question_type": "prohibition",
        "document_type": "statute",
        "source_path": "documents/raw/statutes/약관의 규제에 관한 법률(법률)(제20239호)(20240807).hwpx",
        "question": "약관 조항이 고객에게 부당하게 불리하고 신의성실 원칙을 위반하면 효력이 있나요?",
        "gold_answer": "아니요. 신의성실의 원칙을 위반하여 공정성을 잃은 약관 조항은 무효이다.",
        "gold_citations": ["제6조제1항"],
        "evaluation_mode": "llm_judge",
        "difficulty": "medium",
        "notes": "Prohibition/invalidity question where a yes/no conclusion is useful.",
    },
    {
        "question_type": "obligation",
        "document_type": "contract",
        "source_path": "documents/raw/contracts/주택임대차 표준계약서(원본게시용).pdf",
        "question": "주택임대차 표준계약서에서 임차인은 차임을 언제 지급해야 하나요?",
        "gold_answer": "임차인은 계약에서 정한 지급 시기에 차임을 임대인에게 지급하여야 한다.",
        "gold_citations": ["주택임대차 표준계약서(원본게시용).pdf"],
        "evaluation_mode": "llm_judge",
        "difficulty": "medium",
        "notes": "Contract payment obligation question.",
    },
    {
        "question_type": "condition",
        "document_type": "qna",
        "source_path": "documents/raw/qna/상속_생활법령_QA.pdf",
        "question": "상속에서 배우자는 언제 1순위 또는 2순위 상속인과 함께 상속인이 되나요?",
        "gold_answer": "배우자는 직계비속이 있으면 직계비속과 공동상속인이 되고, 직계비속이 없고 직계존속이 있으면 직계존속과 공동상속인이 된다.",
        "gold_citations": ["상속_생활법령_QA.pdf"],
        "evaluation_mode": "llm_judge",
        "difficulty": "hard",
        "notes": "Conditional guidance question needing rule application.",
    },
    {
        "question_type": "scenario_apply",
        "document_type": "qna",
        "source_path": "documents/raw/qna/신용카드_이용자_생활법령_QA.pdf",
        "question": "신용카드를 분실한 뒤 즉시 카드사에 신고하면 신고 이후 손해는 누가 부담하나요?",
        "gold_answer": "일반적으로 신고 이후에 발생한 손해는 카드사가 부담한다.",
        "gold_citations": ["신용카드_이용자_생활법령_QA.pdf"],
        "evaluation_mode": "llm_judge",
        "difficulty": "medium",
        "notes": "Consumer guidance scenario where vector search can confuse similar finance documents.",
    },
    {
        "question_type": "scenario_apply",
        "document_type": "qna",
        "source_path": "documents/raw/qna/주택임대차_생활법령_QA.pdf",
        "question": "주택 임차인이 대항력을 갖추려면 어떤 두 가지 요건이 필요한가요?",
        "gold_answer": "주택의 인도와 주민등록을 마쳐야 한다.",
        "gold_citations": ["주택임대차_생활법령_QA.pdf"],
        "evaluation_mode": "llm_judge",
        "difficulty": "medium",
        "notes": "Guidance question where document-source retrieval should matter.",
    },
    {
        "question_type": "contract_duty",
        "document_type": "contract",
        "source_path": "documents/raw/contracts/주택임대차 표준계약서(원본게시용).pdf",
        "question": "주택임대차 표준계약서에서 임대인은 임차인이 목적물을 사용할 수 있도록 무엇을 해야 하나요?",
        "gold_answer": "임대인은 임차주택을 임차인에게 인도하고, 임차인이 계약 목적대로 사용ㆍ수익할 수 있도록 해야 한다.",
        "gold_citations": ["주택임대차 표준계약서(원본게시용).pdf"],
        "evaluation_mode": "llm_judge",
        "difficulty": "medium",
        "notes": "Contract duty question; avoids generic document-summary wording.",
    },
    {
        "question_type": "contract_duty",
        "document_type": "contract",
        "source_path": "documents/raw/contracts/상가건물_임대차_표준계약서.pdf",
        "question": "상가건물 표준임대차계약서에서 임차인은 계약 종료 때 어떤 반환 의무를 부담하나요?",
        "gold_answer": "임차인은 임대차가 종료되면 임차 상가건물을 원상으로 회복하여 임대인에게 반환하여야 한다.",
        "gold_citations": ["상가건물_임대차_표준계약서.pdf"],
        "evaluation_mode": "llm_judge",
        "difficulty": "medium",
        "notes": "Contract obligation question for a non-statute PDF source.",
    },
    {
        "question_type": "citation",
        "document_type": "precedent",
        "source_path": "documents/raw/precedents/판례_2010두7031.hwpx",
        "question": "2010두7031 판결은 어떤 행정행위성이 문제 된 사건인가요?",
        "gold_answer": "택지개발사업 시행자의 택지 공급방법 결정 및 통보가 항고소송의 대상이 되는 행정처분인지가 문제 되었다.",
        "gold_citations": ["2010두7031"],
        "evaluation_mode": "llm_judge",
        "difficulty": "hard",
        "notes": "Precedent question where source/case matching should help GraphRAG.",
    },
]


def normalized_question(question: str) -> str:
    return re.sub(r"\s+", "", question.casefold())


def is_generic(question: EvaluationQuestion) -> bool:
    if question.source_path in REMOVED_SIMILAR_SOURCES:
        return True
    if any(pattern in question.question for pattern in GENERIC_PATTERNS):
        return True
    return any(pattern.search(question.question) for pattern in MALFORMED_PATTERNS)


def priority(question: EvaluationQuestion) -> tuple[int, int]:
    if question.id.startswith("eval-focus-"):
        return (-1, 0)
    type_priority = {
        "scenario_apply": 0,
        "condition": 1,
        "exception": 2,
        "citation": 3,
        "comparison": 4,
        "contract_duty": 5,
        "regulatory_guidance": 6,
        "prohibition": 7,
        "obligation": 8,
        "definition_lookup": 9,
        "article_lookup": 10,
        "holding": 11,
    }
    source_priority = 0 if question.id.startswith("eval-") else 1
    return (type_priority.get(question.question_type, 99), source_priority)


def added_question(index: int, row: dict) -> EvaluationQuestion:
    return EvaluationQuestion(
        id=f"eval-focus-{index:03d}",
        relevant_terms=[],
        options=None,
        **row,
    )


def rebalance() -> list[EvaluationQuestion]:
    if not BACKUP_PATH.exists():
        BACKUP_PATH.write_text(CURATED_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    merged = load_questions(BACKUP_PATH) + load_questions(CODEX_PATH)
    merged = [question for question in merged if not is_generic(question)]
    merged.extend(
        added_question(index, row)
        for index, row in enumerate(ADDED_QUESTIONS, start=1)
    )
    merged.sort(key=priority)

    selected = []
    seen_questions = set()
    source_counts: dict[str, int] = {}
    for question in merged:
        key = normalized_question(question.question)
        if key in seen_questions:
            continue
        if source_counts.get(question.source_path, 0) >= PER_SOURCE_LIMIT:
            continue
        selected.append(question)
        seen_questions.add(key)
        source_counts[question.source_path] = source_counts.get(question.source_path, 0) + 1
        if len(selected) >= TARGET_TOTAL:
            break

    return selected


def main() -> None:
    selected = rebalance()
    with open(CURATED_PATH, "w", encoding="utf-8") as file:
        for question in selected:
            file.write(json.dumps(asdict(question), ensure_ascii=False) + "\n")
    sources = {question.source_path for question in selected}
    types: dict[str, int] = {}
    for question in selected:
        types[question.question_type] = types.get(question.question_type, 0) + 1
    print(f"wrote {len(selected)} questions across {len(sources)} sources")
    print(types)


if __name__ == "__main__":
    main()
