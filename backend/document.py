from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Document:
    content: str
    meta: dict = field(default_factory=dict)
    score: float | None = None

    def with_meta(self, **values) -> "Document":
        """Return a copy with updated metadata."""
        return replace(self, meta={**self.meta, **values})

    @property
    def evidence_key(self):
        """Return the stable identity used to merge retrieval results."""
        return self.meta.get("id") or (
            self.meta.get("source_path"),
            self.meta.get("chunk_number"),
        )
