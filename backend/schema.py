def setup_schema(run_write) -> None:
    run_write(
        """
        CREATE CONSTRAINT document_source_path IF NOT EXISTS
        FOR (document:Document)
        REQUIRE document.source_path IS UNIQUE
        """
    )
    run_write(
        """
        CREATE CONSTRAINT chunk_id IF NOT EXISTS
        FOR (chunk:Chunk)
        REQUIRE chunk.id IS UNIQUE
        """
    )
    run_write(
        """
        CREATE CONSTRAINT legal_unit_id IF NOT EXISTS
        FOR (legal_unit:LegalUnit)
        REQUIRE legal_unit.id IS UNIQUE
        """
    )
    run_write(
        """
        CREATE CONSTRAINT definition_id IF NOT EXISTS
        FOR (definition:Definition)
        REQUIRE definition.id IS UNIQUE
        """
    )
    run_write(
        """
        CREATE CONSTRAINT term_id IF NOT EXISTS
        FOR (term:Term)
        REQUIRE term.id IS UNIQUE
        """
    )
    run_write(
        """
        CREATE CONSTRAINT concept_name IF NOT EXISTS
        FOR (concept:Concept)
        REQUIRE concept.name IS UNIQUE
        """
    )

def setup_vector_index(
    run_write,
    run_read,
    index_name: str,
    embedding_dimensions: int,
) -> None:
    if not vector_index_matches_embedding_model(
        run_read,
        index_name,
        embedding_dimensions,
    ):
        run_write(f"DROP INDEX {index_name} IF EXISTS")
    run_write(
        f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (chunk:Chunk)
        ON (chunk.embedding)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {embedding_dimensions},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
    )


def vector_index_matches_embedding_model(
    run_read,
    index_name: str,
    embedding_dimensions: int,
) -> bool:
    records = run_read(
        """
        SHOW VECTOR INDEXES
        YIELD name, options
        WHERE name = $index_name
        RETURN options
        """,
        index_name=index_name,
    )
    if not records:
        return False

    index_config = records[0]["options"].get("indexConfig", {})
    return index_config.get("vector.dimensions") == embedding_dimensions


def vector_index_exists(run_read, index_name: str) -> bool:
    records = run_read(
        """
        SHOW VECTOR INDEXES
        YIELD name
        WHERE name = $index_name
        RETURN count(*) AS count
        """,
        index_name=index_name,
    )
    return bool(records and records[0]["count"])
