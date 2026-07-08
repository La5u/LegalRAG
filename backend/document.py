from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    content: str
    meta: dict = field(default_factory=dict)
    score: float | None = None
