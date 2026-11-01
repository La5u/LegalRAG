import json
import re
import sys
from pathlib import Path
from typing import Any

from backend.document_types import document_type_from_source_path
from backend.loaders import load_document

RAW_DIR = Path("documents/raw")
OUTPUT_PATH = Path("evaluation/generated_candidates.jsonl")
MIN_CHARS = 200

ARTICLE_PATTERN = re.compile(r"제(\d+)조(?:의(\d+))?(?:\(([^)]*)\))?")


def short_name(path: Path) -> str:
    """Return a short human-readable source name."""
    name = path.stem
    name = re.sub(r"\([^)]*\)", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:60]


def extract_articles(text: str) -> list[dict[str, Any]]:
    """Extract article number, title, and the text that follows."""
    articles = []
    lines = text.split("\n")
    current_article = None
    current_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = ARTICLE_PATTERN.match(line)
        if match:
            if current_article:
                current_article["text"] = "\n".join(current_lines)
                articles.append(current_article)
            main = match.group(1)
            sub = match.group(2) or ""
            number = f"제{main}조" + (f"의{sub}" if sub else "")
            title = match.group(3) or ""
            current_article = {"number": number, "title": title, "text": ""}
            current_lines = [line]
        elif current_article:
            current_lines.append(line)
    if current_article:
        current_article["text"] = "\n".join(current_lines)
        articles.append(current_article)
    return articles


def guess_document_type(path: Path) -> str:
    return document_type_from_source_path(str(path))


def make_article_lookup(article: dict, source_path: str, doc_type: str) -> dict | None:
    title = article["title"]
    if not title:
        return None
    question = f"{short_name(Path(source_path))}에서 '{title}'는 몇 조에 규정되어 있나요?"
    return {
        "id": "",
        "source_path": source_path,
        "document_type": doc_type,
        "question_type": "article_lookup",
        "question": question,
        "gold_answer": article["number"],
        "gold_citations": [article["number"]],
        "evaluation_mode": "exact",
        "difficulty": "easy",
        "notes": f"Article lookup for {article['number']}({title}).",
    }


def make_definition_question(
    article: dict, term: str, source_path: str, doc_type: str
) -> dict | None:
    if not term or len(term) < 2:
        return None
    question = f"{short_name(Path(source_path))}에서 '{term}'의 정의는 무엇인가요?"
    answer_text = article["text"][:300] if article["text"] else f"{article['number']} 참조"
    return {
        "id": "",
        "source_path": source_path,
        "document_type": doc_type,
        "question_type": "definition_lookup",
        "question": question,
        "gold_answer": answer_text,
        "gold_citations": [article["number"]],
        "evaluation_mode": "llm_judge",
        "difficulty": "easy",
        "notes": f"Definition of '{term}' from {article['number']}.",
    }


def make_obligation_question(
    article: dict, source_path: str, doc_type: str
) -> dict | None:
    text = article["text"]
    if not text or "하여야" not in text:
        return None
    match = re.search(r"는\s*(.*?)(?:하여야|해야)", text[:200])
    subject = match.group(1)[-20:] if match else short_name(Path(source_path))
    question = f"{short_name(Path(source_path))} {article['number']}에 따라 {subject}의 의무는 무엇인가요?"
    return {
        "id": "",
        "source_path": source_path,
        "document_type": doc_type,
        "question_type": "obligation",
        "question": question,
        "gold_answer": text[:300],
        "gold_citations": [article["number"]],
        "evaluation_mode": "llm_judge",
        "difficulty": "medium",
        "notes": f"Obligation question from {article['number']}.",
    }


def make_yes_no_question(
    article: dict, source_path: str, doc_type: str
) -> dict | None:
    text = article["text"]
    if not text or "하여야" not in text:
        return None
    match = re.search(r"는\s*(.*?)(?:하여야)", text[:150])
    subject = match.group(1)[-20:] if match else None
    if not subject:
        return None
    question = f"{short_name(Path(source_path))} {article['number']}에 따라 {subject}는 의무를 이행하여야 하나요?"
    return {
        "id": "",
        "source_path": source_path,
        "document_type": doc_type,
        "question_type": "yes_no",
        "question": question,
        "gold_answer": "예",
        "gold_citations": [article["number"]],
        "evaluation_mode": "yes_no",
        "difficulty": "easy",
        "notes": f"Yes/no question from {article['number']}.",
    }


def make_exception_question(
    article: dict, source_path: str, doc_type: str
) -> dict | None:
    text = article["text"]
    if not text or "제외" not in text and "아니" not in text and "불구" not in text:
        return None
    question = f"{short_name(Path(source_path))} {article['number']}에서 규정한 예외는 무엇인가요?"
    return {
        "id": "",
        "source_path": source_path,
        "document_type": doc_type,
        "question_type": "exception",
        "question": question,
        "gold_answer": text[:300],
        "gold_citations": [article["number"]],
        "evaluation_mode": "llm_judge",
        "difficulty": "medium",
        "notes": f"Exception question from {article['number']}.",
    }


def make_prohibition_question(
    article: dict, source_path: str, doc_type: str
) -> dict | None:
    text = article["text"]
    if not text or "아니" not in text and "금지" not in text:
        return None
    match = re.search(r"는\s*(.*?)(?:아니|금지)", text[:150])
    subject = match.group(1)[-20:] if match else None
    if not subject:
        return None
    question = f"{short_name(Path(source_path))} {article['number']}에서 금지되는 행위는 무엇인가요?"
    return {
        "id": "",
        "source_path": source_path,
        "document_type": doc_type,
        "question_type": "prohibition",
        "question": question,
        "gold_answer": text[:300],
        "gold_citations": [article["number"]],
        "evaluation_mode": "llm_judge",
        "difficulty": "medium",
        "notes": f"Prohibition question from {article['number']}.",
    }


def make_citation_question(
    article: dict, source_path: str, doc_type: str
) -> dict | None:
    title = article["title"]
    if not title:
        return None
    q = f"{short_name(Path(source_path))}에서 '{title}'에 관한 내용은 몇 조에서 찾을 수 있나요?"
    return {
        "id": "",
        "source_path": source_path,
        "document_type": doc_type,
        "question_type": "citation",
        "question": q,
        "gold_answer": article["number"],
        "gold_citations": [article["number"]],
        "evaluation_mode": "citation",
        "difficulty": "easy",
        "notes": f"Citation question for {article['number']}({title}).",
    }


def find_defined_terms(text: str) -> list[str]:
    terms = []
    for m in re.finditer(r"[「\"']([^」\"']{2,30})[」\"'](?:이란|란|이라\s*함은|라\s*함은)", text):
        terms.append(m.group(1))
    for m in re.finditer(r"“([^”]{2,30})”\s*(?:이란|란)", text):
        terms.append(m.group(1))
    return terms


def generate_for_statute(
    text: str, source_path: str
) -> list[dict]:
    items = []
    articles = extract_articles(text)
    doc_type = "statute"

    seen_terms = set()
    for article in articles:
        if not article["text"]:
            continue
        text_l = len(article["text"])
        if text_l < MIN_CHARS:
            continue

        terms = find_defined_terms(article["text"])
        for term in terms:
            if term not in seen_terms:
                seen_terms.add(term)
                q = make_definition_question(article, term, source_path, doc_type)
                if q:
                    items.append(q)

        al = make_article_lookup(article, source_path, doc_type)
        if al:
            items.append(al)

        obl = make_obligation_question(article, source_path, doc_type)
        if obl:
            items.append(obl)

        yn = make_yes_no_question(article, source_path, doc_type)
        if yn:
            items.append(yn)

        exc = make_exception_question(article, source_path, doc_type)
        if exc:
            items.append(exc)

        pro = make_prohibition_question(article, source_path, doc_type)
        if pro:
            items.append(pro)

    return items


def generate_for_precedent(
    text: str, source_path: str
) -> list[dict]:
    items = []
    doc_type = "precedent"
    name = short_name(Path(source_path))

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    reason_start = -1
    for i, line in enumerate(lines):
        if line == "이유" or line.startswith("이유"):
            reason_start = i
            break
    holding_text = "\n".join(lines[reason_start + 1 : reason_start + 6]) if reason_start >= 0 else text[:500]
    case_id = Path(source_path).stem[:20]

    q = {
        "id": "",
        "source_path": source_path,
        "document_type": doc_type,
        "question_type": "holding",
        "question": f"대법원 {name} 판결의 주요 쟁점 또는 판시 사항은 무엇인가요?",
        "gold_answer": holding_text[:400],
        "gold_citations": [case_id],
        "evaluation_mode": "llm_judge",
        "difficulty": "hard",
        "notes": f"Precedent holding question for {case_id}.",
    }
    items.append(q)

    if len(lines) > 10:
        q2 = {
            "id": "",
            "source_path": source_path,
            "document_type": doc_type,
            "question_type": "citation",
            "question": f"대법원 {name} 사건의 사건번호는 무엇인가요?",
            "gold_answer": case_id,
            "gold_citations": [case_id],
            "evaluation_mode": "exact",
            "difficulty": "easy",
            "notes": f"Citation for {case_id}.",
        }
        items.append(q2)

    return items


def generate_for_regulation(
    text: str, source_path: str
) -> list[dict]:
    return generate_for_statute(text, source_path)


def generate_for_guidance(
    text: str, source_path: str
) -> list[dict]:
    items = []
    doc_type = "guidance"
    name = short_name(Path(source_path))
    sentences = re.split(r"(?<=[.다])\s+", text[:2000])
    first_sentence = sentences[0] if sentences else text[:300]

    q = {
        "id": "",
        "source_path": source_path,
        "document_type": doc_type,
        "question_type": "regulatory_guidance",
        "question": f"{name}의 주요 내용은 무엇인가요?",
        "gold_answer": first_sentence[:400],
        "gold_citations": [Path(source_path).name],
        "evaluation_mode": "llm_judge",
        "difficulty": "medium",
        "notes": f"Guidance summary question for {Path(source_path).name}.",
    }
    items.append(q)

    match = re.search(r"(?:목적|purpose).*?(?:[.。])", text[:1000])
    if match:
        q2 = {
            "id": "",
            "source_path": source_path,
            "document_type": doc_type,
            "question_type": "regulatory_guidance",
            "question": f"{name}의 발간 목적은 무엇인가요?",
            "gold_answer": match.group(0)[:300],
            "gold_citations": [Path(source_path).name],
            "evaluation_mode": "llm_judge",
            "difficulty": "medium",
            "notes": f"Purpose question for {Path(source_path).name}.",
        }
        items.append(q2)

    return items


def generate_for_contract(
    text: str, source_path: str
) -> list[dict]:
    items = []
    doc_type = "contract"
    name = short_name(Path(source_path))

    articles = extract_articles(text)
    for article in articles[:5]:
        if not article["text"]:
            continue
        if "하여야" in article["text"]:
            q = make_obligation_question(article, source_path, doc_type)
            if q:
                items.append(q)
        al = make_article_lookup(article, source_path, doc_type)
        if al:
            items.append(al)

    if not items:
        q = {
            "id": "",
            "source_path": source_path,
            "document_type": doc_type,
            "question_type": "contract_duty",
            "question": f"{name} 계약서에서 임대인의 주요 의무는 무엇인가요?" if "임대" in name or "주택" in name else f"{name} 계약서에서 당사자의 주요 의무는 무엇인가요?",
            "gold_answer": text[:400],
            "gold_citations": [Path(source_path).name],
            "evaluation_mode": "llm_judge",
            "difficulty": "medium",
            "notes": f"Contract duty question for {Path(source_path).name}.",
        }
        items.append(q)

    return items


def generate_for_qna(
    text: str, source_path: str
) -> list[dict]:
    items = []
    doc_type = "qna"
    name = short_name(Path(source_path))

    sentences = re.split(r"(?<=[.다])\s+", text[:2000])
    qa_pairs = []
    current_q = None
    current_a_lines = []
    for s in sentences[:30]:
        s = s.strip()
        if not s:
            continue
        if "?" in s or "?" in s:
            if current_q and current_a_lines:
                qa_pairs.append((current_q, " ".join(current_a_lines)))
            current_q = s[:100]
            current_a_lines = []
        elif current_q:
            current_a_lines.append(s)
    if current_q and current_a_lines:
        qa_pairs.append((current_q, " ".join(current_a_lines)))

    for qi, (q_text, a_text) in enumerate(qa_pairs[:3]):
        items.append({
            "id": "",
            "source_path": source_path,
            "document_type": doc_type,
            "question_type": "scenario_apply",
            "question": q_text,
            "gold_answer": a_text[:400],
            "gold_citations": [Path(source_path).name],
            "evaluation_mode": "llm_judge",
            "difficulty": "medium",
            "notes": f"Q&A pair {qi+1} from {Path(source_path).name}.",
        })

    if not items:
        items.append({
            "id": "",
            "source_path": source_path,
            "document_type": doc_type,
            "question_type": "regulatory_guidance",
            "question": f"{name} 안내서의 주요 내용은 무엇인가요?",
            "gold_answer": text[:400],
            "gold_citations": [Path(source_path).name],
            "evaluation_mode": "llm_judge",
            "difficulty": "medium",
            "notes": f"Guidance question for {Path(source_path).name}.",
        })

    return items


def generate_for_document(source_path: Path) -> list[dict]:
    try:
        text = load_document(source_path)
    except Exception:
        return []
    if not text or len(text) < MIN_CHARS:
        return []

    doc_type = guess_document_type(source_path)
    s_path = str(source_path)

    if doc_type in ("statute",):
        items = generate_for_statute(text, s_path)
    elif doc_type == "regulation":
        items = generate_for_regulation(text, s_path)
    elif doc_type == "precedent":
        items = generate_for_precedent(text, s_path)
    elif doc_type == "guidance":
        items = generate_for_guidance(text, s_path)
    elif doc_type == "contract":
        items = generate_for_contract(text, s_path)
    elif doc_type == "qna":
        items = generate_for_qna(text, s_path)
    else:
        items = generate_for_statute(text, s_path)

    for item in items:
        item["document_type"] = doc_type
    return items


def main() -> None:
    source_dirs = sorted(RAW_DIR.iterdir())
    all_candidates: list[dict] = []
    seen_stems: set[str] = set()

    SKIP_DIRS = {"excluded"}
    SKIP_SUFFIXES = {".hwp", ".zip"}

    for subdir in source_dirs:
        if not subdir.is_dir() or subdir.name in SKIP_DIRS:
            continue
        for path in sorted(subdir.glob("**/*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in SKIP_SUFFIXES:
                continue
            if suffix not in (".hwpx", ".pdf", ".txt"):
                continue
            stem = path.stem
            if subdir.name in ("statutes", "regulations", "constitutional_decisions"):
                if suffix == ".pdf":
                    hwpx_path = path.with_suffix(".hwpx")
                    if hwpx_path.exists():
                        continue
            if stem in seen_stems:
                continue
            seen_stems.add(stem)
            print(f"  {path}", file=sys.stderr)
            items = generate_for_document(path)
            all_candidates.extend(items)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in all_candidates:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Generated {len(all_candidates)} candidates from {len(seen_stems)} sources -> {OUTPUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
