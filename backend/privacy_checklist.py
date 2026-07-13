"""Deterministic checklist for fact-based Korean privacy-law questions."""

from backend.document import Document


PRIVACY_TERMS = ("개인정보", "정보주체", "개인정보처리자", "가명정보")
SCENARIO_TERMS = (
    "가능", "적법", "위반", "없이", "되나요", "할 수", "컴플라이언스", "체크리스트",
)
CHECKS = (
    ("처리 대상과 행위", ("개인정보", "수집", "이용", "처리")),
    ("처리의 법적 근거", ("동의", "법률", "계약", "정당한 이익")),
    ("고지 및 투명성", ("알리", "고지", "통지", "처리방침")),
    ("제3자 제공·위탁", ("제3자", "제공", "위탁", "수탁")),
    ("보유기간 및 파기", ("보유기간", "보존", "파기", "삭제")),
    ("안전성 확보조치", ("안전성", "보호조치", "암호화", "접근권한")),
    ("정보주체의 권리", ("열람", "정정", "삭제", "처리정지")),
    ("국외 이전", ("국외", "해외", "이전")),
)


def privacy_checklist(query: str, evidence: list[Document]) -> dict | None:
    """Return an issue matrix only for privacy compliance scenarios.

    ``rule_found`` means retrieved evidence addresses the issue. It does not
    mean the user's conduct is compliant; that requires the missing facts
    listed by the checklist.
    """
    if not any(term in query for term in PRIVACY_TERMS):
        return None
    if not any(term in query for term in SCENARIO_TERMS):
        return None

    rows = []
    for label, terms in CHECKS:
        matches = [document for document in evidence if any(term in document.content for term in terms)]
        fact_stated = any(term in query for term in terms)
        if fact_stated or label in {"처리 대상과 행위", "처리의 법적 근거"}:
            rows.append({
                "issue": label,
                "rule_status": "found" if matches else "not_found",
                "fact_status": "stated" if fact_stated else "missing",
                "evidence": _reference(matches[0]) if matches else None,
            })

    missing = [row["issue"] for row in rows if row["fact_status"] == "missing"]
    return {
        "workflow": "privacy_compliance",
        "checks": rows,
        "missing_facts": missing,
        "result": "needs_facts" if missing else "ready_for_legal_analysis",
    }


def _reference(document: Document) -> dict:
    return {
        "document": document.meta.get("document_name"),
        "chunk": document.meta.get("chunk_number"),
        "article": document.meta.get("article_number"),
    }
