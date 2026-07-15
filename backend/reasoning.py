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
GRAPH_FACT_PREFIXES = ("Graph fact:", "그래프 사실:", "그래프 관계 후보:")
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
        "important_evidence": important_evidence(
            prohibition_facts,
            permission_facts,
            obligation_facts,
            exception_facts,
            condition_facts,
            evidence,
        ),
        "definitions_checked": trace_items(categories["definitions"], limit=2),
        "prohibitions_checked": trace_items(prohibition_facts, limit=2),
        "permissions_checked": trace_items(permission_facts, limit=2),
        "obligations_checked": trace_items(obligation_facts, limit=2),
        "exceptions_checked": trace_items(exception_facts, limit=2),
        "conditions_checked": trace_items(condition_facts, limit=4),
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
            if not document.content.startswith(GRAPH_FACT_PREFIXES)
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


def _is_definition_fact(fact): return fact.meta.get("graph_relation") in DEFINITION_RELATIONS or "DEFINED_AS" in fact.content
def _is_relation_fact(fact): return "->" in fact.content
def _is_permission_fact(fact): return fact.meta.get("graph_relation") in PERMISSION_RELATIONS or contains_permission(fact.content)
def _is_obligation_fact(fact): return fact.meta.get("graph_relation") in OBLIGATION_RELATIONS or contains_any(fact.content, OBLIGATION_TEXT_TERMS)
def _is_exception_fact(fact): return any(term in fact.content for term in EXCEPTION_TERMS)
def _is_condition_fact(fact): return any(term in fact.content.casefold() for term in CONDITION_TERMS)


def categorize_evidence(query: str, evidence: list[Document]) -> dict[str, list[Document]]:
    query_terms = relevant_query_terms(query)
    graph_facts = [d for d in evidence if d.content.startswith(GRAPH_FACT_PREFIXES)]
    text_facts = [d for d in evidence if not d.content.startswith(GRAPH_FACT_PREFIXES)]
    sources = {"graph": graph_facts, "text": text_facts, "both": graph_facts + text_facts}

    specs = [
        ("definitions", "graph", _is_definition_fact, False),
        ("relation_facts", "graph", _is_relation_fact, False),
        ("prohibition_facts", "both", is_prohibition_fact, True),
        ("permission_facts", "both", _is_permission_fact, True),
        ("obligation_facts", "both", _is_obligation_fact, True),
        ("exception_facts", "text", _is_exception_fact, True),
        ("condition_facts", "text", _is_condition_fact, True),
    ]
    categories = {}
    for key, src, pred, needs_relevance in specs:
        facts = [f for f in sources[src] if pred(f)]
        if needs_relevance:
            facts = [f for f in facts if is_relevant_fact(f, query_terms)]
            facts = sort_by_relevance(facts, query_terms)
        categories[key] = facts
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
    def rule(*args):
        return {"answer_type": args[0], "reason": args[1]}

    if query_type == "article_lookup" and condition_facts:
        return rule("article_text", "명시된 조문의 본문을 근거로 답변합니다.")
    if prohibition_facts:
        return rule("no", "A prohibition was found in the retrieved evidence.")
    if (
        exception_facts
        and "그러하지 아니하다" in exception_facts[0].content
        and ("취소" in query or "수 있" in query or "되나요" in query)
    ):
        return rule("no", "A directly relevant exception removes the ordinary rule.")
    if query_type in {"condition", "permission", "scenario_apply"} and permission_facts:
        return rule("yes", "A permission was found in the retrieved evidence.")
    if query_type == "obligation" and obligation_facts:
        return rule("yes", "A duty was found in the retrieved evidence.")
    if exception_facts:
        return rule("conditional", "An exception or proviso was found in the retrieved evidence.")
    if condition_facts and query_type in {"condition", "scenario_apply"}:
        return rule("conditional", "A condition was found and must be applied to answer the question.")
    if query_type == "obligation" and relations & OBLIGATION_RELATIONS:
        return rule("yes", "A duty relation was found in the graph evidence.")
    if query_type == "prohibition" and relations & PROHIBITION_RELATIONS:
        return rule("no", "A prohibition relation was found in the graph evidence.")
    if query_type == "permission" and relations & PERMISSION_RELATIONS:
        return rule("yes", "A permission relation was found in the graph evidence.")
    if relation_facts:
        return rule("conditional", "Legal relations were found, but they do not directly resolve the question type.")
    return rule("not_answered", "No symbolic rule relation was found in the retrieved evidence.")


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
