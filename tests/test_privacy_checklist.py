from backend.document import Document
from backend.privacy_checklist import privacy_checklist


def test_privacy_scenario_builds_a_grounded_issue_matrix():
    evidence = [
        Document(
            "개인정보처리자는 정보주체의 동의를 받아 개인정보를 수집할 수 있다.",
            {"document_name": "개인정보 보호법.hwpx", "chunk_number": 1, "article_number": "제15조"},
        )
    ]
    result = privacy_checklist("개인정보를 동의 없이 수집할 수 있나요?", evidence)
    assert result["workflow"] == "privacy_compliance"
    assert result["checks"][1]["issue"] == "처리의 법적 근거"
    assert result["checks"][1]["rule_status"] == "found"
    assert result["checks"][1]["fact_status"] == "stated"


def test_definition_question_does_not_start_compliance_workflow():
    assert privacy_checklist("개인정보란 무엇인가요?", []) is None
