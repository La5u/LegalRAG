from backend.document import Document
from backend.privacy_checklist import privacy_checklist
from backend.routing import classify_query
from backend.text import query_terms


OBLIGATION_RELATIONS = {"HAS_DUTY", "MUST_NOTIFY"}
PERMISSION_RELATIONS = {"MAY", "MAY_REQUEST"}
PROHIBITION_RELATIONS = {"PROHIBITS"}
DEFINITION_RELATIONS = {"DEFINED_AS"}
EXCEPTION_TERMS = ("다만", "예외", "제외", "그러하지 아니하다", "EXCEPTION_TO")
CONDITION_TERMS = ("경우", "때", "요건", "조건", "unless", "when", "if")
PROHIBITION_TEXT_TERMS = (
    "해서는 안",
    "하여서는 안",
    "해서는 아니",
    "하여서는 아니",
    "할 수 없다",
    "할 수 없습니다",
    "금지",
)
PERMISSION_TEXT_TERMS = ("할 수 있다", "할 수 있습니다", "수집할 수 있으며")
OBLIGATION_TEXT_TERMS = ("하여야 한다", "해야 한다", "하여야 합니다", "해야 합니다")


def build_symbolic_trace(query: str, evidence: list[Document]) -> dict:
    route = classify_query(query)
    categories = categorize_evidence(query, evidence)
    relation_facts = categories["relation_facts"]
    prohibition_facts = categories["prohibition_facts"]
    permission_facts = categories["permission_facts"]
    obligation_facts = categories["obligation_facts"]
    exception_facts = categories["exception_facts"]
    condition_facts = categories["condition_facts"]
    conclusion = classify_conclusion(
        query,
        route.kind,
        relation_facts,
        prohibition_facts,
        permission_facts,
        obligation_facts,
        exception_facts,
        condition_facts,
    )

    trace = {
        "query_type": route.kind,
        "summary": trace_summary(route.kind, conclusion, evidence),
        "important_evidence": important_evidence(
            prohibition_facts,
            permission_facts,
            obligation_facts,
            exception_facts,
            condition_facts,
            evidence,
        ),
        "matched_concepts": matched_terms(query),
        "definitions_checked": trace_items(categories["definitions"], limit=2),
        "legal_relations_checked": trace_items(relation_facts, limit=4),
        "prohibitions_checked": trace_items(prohibition_facts, limit=2),
        "permissions_checked": trace_items(permission_facts, limit=2),
        "obligations_checked": trace_items(obligation_facts, limit=2),
        "exceptions_checked": trace_items(exception_facts, limit=2),
        "conditions_checked": trace_items(condition_facts, limit=4),
        "supporting_chunks": supporting_chunks(evidence, limit=6),
        "reasoning_steps": reasoning_steps(
            categories["definitions"],
            relation_facts,
            prohibition_facts,
            permission_facts,
            obligation_facts,
            exception_facts,
            condition_facts,
        ),
        "conclusion": conclusion,
    }
    checklist = privacy_checklist(query, evidence)
    if checklist:
        trace["workflow"] = checklist["workflow"]
        trace["privacy_checklist"] = checklist["checks"]
        trace["missing_facts"] = checklist["missing_facts"]
        trace["conclusion"] = {
            "answer_type": checklist["result"],
            "reason": "필수 쟁점의 근거와 누락 정보를 확인했습니다.",
        }
    return trace


def trace_summary(query_type: str, conclusion: dict, evidence: list[Document]) -> list[str]:
    steps = [f"질문 유형: {query_type}"]
    if query_type == "article_lookup":
        articles = list(
            dict.fromkeys(
                document.meta.get("article_number")
                for document in evidence
                if document.meta.get("article_number")
            )
        )
        if articles:
            steps.append(f"명시된 조문을 확인함: {', '.join(articles[:3])}")
    steps.append(
        f"결론 유형: {conclusion['answer_type']} - {conclusion['reason']}"
    )
    return steps


def important_evidence(
    prohibition_facts: list[Document],
    permission_facts: list[Document],
    obligation_facts: list[Document],
    exception_facts: list[Document],
    condition_facts: list[Document],
    evidence: list[Document],
) -> list[dict]:
    selected = (
        prohibition_facts[:2]
        + permission_facts[:2]
        + obligation_facts[:2]
        + exception_facts[:1]
        + condition_facts[:4]
    )
    if not selected:
        selected = [
            document
            for document in evidence
            if not document.content.startswith(("Graph fact:", "그래프 사실:", "그래프 관계 후보:"))
        ][:3]
    selected = unique_documents(selected)
    return trace_items(selected, limit=6)


def unique_documents(documents: list[Document]) -> list[Document]:
    unique = []
    seen = set()
    for document in documents:
        key = (document.meta.get("document_name"), document.meta.get("chunk_number"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(document)
    return unique


def categorize_evidence(query: str, evidence: list[Document]) -> dict[str, list[Document]]:
    query_terms = relevant_query_terms(query)
    graph_facts = [
        document
        for document in evidence
        if document.content.startswith(("Graph fact:", "그래프 사실:", "그래프 관계 후보:"))
    ]
    text_facts = [
        document
        for document in evidence
        if not document.content.startswith(("Graph fact:", "그래프 사실:", "그래프 관계 후보:"))
    ]
    categories = {
        "definitions": [
            fact
            for fact in graph_facts
            if fact.meta.get("graph_relation") in DEFINITION_RELATIONS
            or "DEFINED_AS" in fact.content
        ],
        "relation_facts": [
            fact
            for fact in graph_facts
            if "->" in fact.content
        ],
        "prohibition_facts": [
            fact
            for fact in graph_facts + text_facts
            if is_prohibition_fact(fact)
            if is_relevant_fact(fact, query_terms)
        ],
        "permission_facts": [
            fact
            for fact in graph_facts + text_facts
            if fact.meta.get("graph_relation") in PERMISSION_RELATIONS
            or contains_permission(fact.content)
            if is_relevant_fact(fact, query_terms)
        ],
        "obligation_facts": [
            fact
            for fact in graph_facts + text_facts
            if fact.meta.get("graph_relation") in OBLIGATION_RELATIONS
            or contains_any(fact.content, OBLIGATION_TEXT_TERMS)
            if is_relevant_fact(fact, query_terms)
        ],
        "exception_facts": [
            fact
            for fact in text_facts
            if any(term in fact.content for term in EXCEPTION_TERMS)
            if is_relevant_fact(fact, query_terms)
        ],
        "condition_facts": [
            fact
            for fact in text_facts
            if any(term in fact.content.casefold() for term in CONDITION_TERMS)
            if is_relevant_fact(fact, query_terms)
        ],
    }
    for key in (
        "prohibition_facts",
        "permission_facts",
        "obligation_facts",
        "exception_facts",
        "condition_facts",
    ):
        categories[key] = sort_by_relevance(categories[key], query_terms)
    return categories


def classify_conclusion(
    query: str,
    query_type: str,
    relation_facts: list[Document],
    prohibition_facts: list[Document],
    permission_facts: list[Document],
    obligation_facts: list[Document],
    exception_facts: list[Document],
    condition_facts: list[Document],
) -> dict:
    relations = {
        fact.meta.get("graph_relation")
        for fact in relation_facts
        if fact.meta.get("graph_relation")
    }
    if query_type == "article_lookup" and condition_facts:
        return {
            "answer_type": "article_text",
            "reason": "명시된 조문의 본문을 근거로 답변합니다.",
        }
    if prohibition_facts:
        return {
            "answer_type": "no",
            "reason": "A prohibition was found in the retrieved evidence.",
        }
    if (
        exception_facts
        and "그러하지 아니하다" in exception_facts[0].content
        and ("취소" in query or "수 있" in query or "되나요" in query)
    ):
        return {
            "answer_type": "no",
            "reason": "A directly relevant exception removes the ordinary rule.",
        }
    if query_type in {"condition", "permission", "scenario_apply"} and permission_facts:
        return {
            "answer_type": "yes",
            "reason": "A permission was found in the retrieved evidence.",
        }
    if query_type == "obligation" and obligation_facts:
        return {
            "answer_type": "yes",
            "reason": "A duty was found in the retrieved evidence.",
        }
    if exception_facts:
        return {
            "answer_type": "conditional",
            "reason": "An exception or proviso was found in the retrieved evidence.",
        }
    if condition_facts and query_type in {"condition", "scenario_apply"}:
        return {
            "answer_type": "conditional",
            "reason": "A condition was found and must be applied to answer the question.",
        }
    if query_type == "obligation" and relations & OBLIGATION_RELATIONS:
        return {
            "answer_type": "yes",
            "reason": "A duty relation was found in the graph evidence.",
        }
    if query_type == "prohibition" and relations & PROHIBITION_RELATIONS:
        return {
            "answer_type": "no",
            "reason": "A prohibition relation was found in the graph evidence.",
        }
    if query_type == "permission" and relations & PERMISSION_RELATIONS:
        return {
            "answer_type": "yes",
            "reason": "A permission relation was found in the graph evidence.",
        }
    if relation_facts:
        return {
            "answer_type": "conditional",
            "reason": "Legal relations were found, but they do not directly resolve the question type.",
        }
    return {
        "answer_type": "not_answered",
        "reason": "No symbolic rule relation was found in the retrieved evidence.",
    }


def reasoning_steps(
    definitions: list[Document],
    relation_facts: list[Document],
    prohibition_facts: list[Document],
    permission_facts: list[Document],
    obligation_facts: list[Document],
    exception_facts: list[Document],
    condition_facts: list[Document],
) -> list[str]:
    steps = []
    if definitions:
        steps.append(f"Checked {len(definitions)} definition fact(s).")
    else:
        steps.append("No definition fact was found in the retrieved graph evidence.")
    if relation_facts:
        steps.append(f"Checked {len(relation_facts)} legal relation fact(s).")
    else:
        steps.append("No duty, permission, prohibition, citation, or definition relation was found.")
    if prohibition_facts:
        steps.append(f"Found {len(prohibition_facts)} prohibition candidate(s).")
    if permission_facts:
        steps.append(f"Found {len(permission_facts)} permission candidate(s).")
    if obligation_facts:
        steps.append(f"Found {len(obligation_facts)} obligation candidate(s).")
    if exception_facts:
        steps.append(f"Checked {len(exception_facts)} exception/proviso candidate(s).")
    else:
        steps.append("No exception/proviso candidate was found.")
    if condition_facts:
        steps.append(f"Checked {len(condition_facts)} condition candidate(s).")
    else:
        steps.append("No condition candidate was found.")
    return steps


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def contains_permission(text: str) -> bool:
    return contains_any(text, PERMISSION_TEXT_TERMS) and not contains_any(
        text,
        PROHIBITION_TEXT_TERMS,
    )


def is_prohibition_fact(document: Document) -> bool:
    content = document.content
    if "-MAY" in content or contains_any(content, PERMISSION_TEXT_TERMS):
        return False
    return (
        document.meta.get("graph_relation") in PROHIBITION_RELATIONS
        or contains_any(content, PROHIBITION_TEXT_TERMS)
    )


def relevant_query_terms(query: str) -> list[str]:
    ignored = {"무엇", "어떤", "수", "있나요", "되나요", "없이", "하면", "위한"}
    return query_terms(
        query,
        ignored_terms=ignored,
        min_length=2,
        strip_particles=True,
    )


def is_relevant_fact(document: Document, query_terms: list[str]) -> bool:
    if not query_terms:
        return True
    text = document.content
    if not matches_required_scenario(text, query_terms):
        return False
    if any(term.startswith("수집") for term in query_terms) and "수집" not in text:
        return False
    score = sum(1 for term in query_terms if term in text)
    return score >= min(2, len(query_terms))


def matches_required_scenario(text: str, query_terms: list[str]) -> bool:
    if "가명정보" in query_terms and "가명정보" not in text:
        return False
    if any(term.startswith("개인정보") for term in query_terms) and "계약" in query_terms:
        return "개인정보" in text and ("계약" in text or "이행" in text)
    if "미성년자" in query_terms:
        return "미성년자" in text or "법정대리인" in text
    return True


def sort_by_relevance(
    documents: list[Document],
    query_terms: list[str],
) -> list[Document]:
    return sorted(
        documents,
        key=lambda document: relevance_score(document, query_terms),
        reverse=True,
    )


def relevance_score(document: Document, query_terms: list[str]) -> int:
    return sum(1 for term in query_terms if term in document.content)


def matched_terms(query: str) -> list[str]:
    return query_terms(query, ignored_terms={"what", "which", "where"}, max_terms=12)


def supporting_chunks(evidence: list[Document], limit: int) -> list[str]:
    chunks = []
    for document in evidence:
        if document.content.startswith(("Graph fact:", "그래프 사실:", "그래프 관계 후보:")):
            continue
        label = f"{document.meta.get('document_name')}#{document.meta.get('chunk_number')}"
        chunks.append(label)
        if len(chunks) == limit:
            break
    return chunks


def trace_items(documents: list[Document], limit: int) -> list[dict]:
    items = []
    for document in documents[:limit]:
        items.append(
            {
                "document": document.meta.get("document_name"),
                "chunk": document.meta.get("chunk_number"),
                "relation": document.meta.get("graph_relation"),
                "article": document.meta.get("article_number"),
                "text": " ".join(document.content.split())[:240],
            }
        )
    return items
