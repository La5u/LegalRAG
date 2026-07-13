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
        CREATE INDEX document_logical_id IF NOT EXISTS
        FOR (document:Document)
        ON (document.logical_id)
        """
    )
    run_write(
        """
        CREATE INDEX document_type IF NOT EXISTS
        FOR (document:Document)
        ON (document.document_type)
        """
    )
    run_write(
        """
        CREATE INDEX chunk_source_path IF NOT EXISTS
        FOR (chunk:Chunk)
        ON (chunk.source_path)
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
    run_write(
        """
        CREATE CONSTRAINT case_source_path IF NOT EXISTS
        FOR (case_node:Case)
        REQUIRE case_node.source_path IS UNIQUE
        """
    )
    run_write(
        """
        CREATE CONSTRAINT holding_id IF NOT EXISTS
        FOR (holding:Holding)
        REQUIRE holding.id IS UNIQUE
        """
    )
    run_write(
        """
        CREATE CONSTRAINT citation_id IF NOT EXISTS
        FOR (citation:Citation)
        REQUIRE citation.id IS UNIQUE
        """
    )

def setup_vector_index(
    run_write,
    run_read,
    index_name: str,
    embedding_dimensions: int,
) -> None:
    existing_dimensions = vector_index_dimensions(run_read, index_name)
    if existing_dimensions == embedding_dimensions:
        return
    if existing_dimensions is not None:
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


def vector_index_dimensions(
    run_read,
    index_name: str,
) -> int | None:
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
        return None

    index_config = records[0]["options"].get("indexConfig", {})
    return index_config.get("vector.dimensions")


def vector_index_exists(run_read, index_name: str) -> bool:
    records = run_read(
        """
        SHOW VECTOR INDEXES
        YIELD name
        WHERE name = $index_name
        RETURN name
        """,
        index_name=index_name,
    )
    return bool(records)
