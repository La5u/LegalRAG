from evaluation.generate_candidates import (
    generate_for_regulation,
    make_definition_question,
    make_exception_question,
)


def article(text: str) -> dict:
    return {"number": "제2조", "title": "정의", "text": text}


def test_definition_gold_uses_requested_definition_line():
    result = make_definition_question(
        article("1. 첫째란 첫 뜻을 말한다.\n2. 둘째란 둘째 뜻을 말한다."),
        "둘째",
        "documents/raw/statutes/법.hwpx",
        "statute",
    )

    assert result is not None
    assert result["gold_answer"] == "2. 둘째란 둘째 뜻을 말한다."


def test_exception_generator_rejects_ordinary_negative_wording():
    result = make_exception_question(
        article("종사자와 직접 대면하지 아니하고 자동화된 거래를 말한다."),
        "documents/raw/statutes/법.hwpx",
        "statute",
    )

    assert result is None


def test_regulation_candidates_keep_regulation_document_type():
    text = "제2조(정의)\n" + "용어란 뜻을 말한다. " * 20

    results = generate_for_regulation(
        text,
        "documents/raw/regulations/규정.hwpx",
    )

    assert results
    assert all(result["document_type"] == "regulation" for result in results)
