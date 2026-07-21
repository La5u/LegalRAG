from pathlib import Path


STRUCTURED_DOCUMENT_TYPES = {"statute", "regulation", "contract"}
PRECEDENT_DOCUMENT_TYPES = {"precedent", "constitutional_decision"}


def document_type_from_source_path(source_path: str) -> str:
    parts = Path(source_path).parts
    if "statutes" in parts:
        return "statute"
    if "regulations" in parts:
        return "regulation"
    if "contracts" in parts:
        return "contract"
    if "precedents" in parts:
        return "precedent"
    if "constitutional_decisions" in parts:
        return "constitutional_decision"
    if "guidance" in parts:
        return "guidance"
    if "qna" in parts:
        return "qna"
    return "unknown"


def has_structural_legal_units(source_path: str) -> bool:
    return document_type_from_source_path(source_path) in STRUCTURED_DOCUMENT_TYPES


def is_precedent_document(source_path: str) -> bool:
    return document_type_from_source_path(source_path) in PRECEDENT_DOCUMENT_TYPES
