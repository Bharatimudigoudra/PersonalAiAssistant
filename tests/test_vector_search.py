"""
Vector Store Diagnostic Test.

Purpose
-------
Diagnose the complete embedding -> Chroma vector-search path.

Checks:
1. Project import path
2. Configuration
3. ChromaDB initialization
4. ChromaDB health
5. Collection metadata
6. Stored document count
7. Stored document content
8. Embedding model
9. Embedding dimension
10. Semantic vector search
11. Search result quality indicators

This is a diagnostic script only.
It does NOT modify or delete the vector database.
"""

from __future__ import annotations

import sys
from pathlib import Path


# ============================================================================
# PROJECT PATH
# ============================================================================

# When executing:
#
#     python tests\test_vector_search.py
#
# Python starts with "tests" as the script directory.
# Add the project root so "app" can be imported reliably.

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# APPLICATION IMPORTS
# ============================================================================

from app.core.config import embedding as embedding_config
from app.core.config import rag
from app.core.config import vectorstore
from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.vectorstore_service import VectorStoreService


# ============================================================================
# DISPLAY HELPERS
# ============================================================================


def section(title: str) -> None:
    """Print a formatted test section."""

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def subsection(title: str) -> None:
    """Print a smaller subsection."""

    print()
    print("-" * 80)
    print(title)
    print("-" * 80)


# ============================================================================
# SAFE VALUE HELPERS
# ============================================================================


def safe_length(value) -> int:
    """Return length without crashing on None."""

    if value is None:
        return 0

    try:
        return len(value)
    except TypeError:
        return 0


def truncate(text: object, limit: int = 1000) -> str:
    """Convert a value to text and truncate it for terminal output."""

    if text is None:
        return ""

    value = str(text)

    if len(value) <= limit:
        return value

    return value[:limit] + "\n...[TRUNCATED]..."


# ============================================================================
# CONFIGURATION TEST
# ============================================================================


def test_configuration() -> None:
    """Display the configuration used by this diagnostic."""

    section("[1] CONFIGURATION")

    print("Project root       :", PROJECT_ROOT)
    print("Vector provider    :", vectorstore.provider)
    print(
        "Persist directory  :",
        vectorstore.persist_directory,
    )
    print(
        "Collection name    :",
        vectorstore.collection_name,
    )

    print()
    print("Embedding provider :", embedding_config.provider)
    print(
        "Embedding model    :",
        embedding_config.model_name,
    )
    print(
        "Embedding device   :",
        embedding_config.device,
    )
    print(
        "Normalize          :",
        embedding_config.normalize_embeddings,
    )

    print()
    print("RAG top_k          :", rag.top_k)
    print(
        "Similarity threshold:",
        rag.similarity_threshold,
    )


# ============================================================================
# VECTOR STORE TEST
# ============================================================================


def initialize_vector_store() -> VectorStoreService:
    """Create and validate the vector-store service."""

    section("[2] VECTOR STORE INITIALIZATION")

    service = VectorStoreService()

    print(
        "Service            :",
        service.__class__.__name__,
    )

    print(
        "Store              :",
        service.store.__class__.__name__,
    )

    return service


# ============================================================================
# HEALTH TEST
# ============================================================================


def test_health(service: VectorStoreService) -> None:
    """Check vector-store health."""

    section("[3] VECTOR STORE HEALTH")

    healthy = service.health_check()

    print("Health             :", healthy)

    if not healthy:
        raise RuntimeError(
            "Vector store health check FAILED."
        )

    print("Status             : PASS")


# ============================================================================
# COLLECTION TEST
# ============================================================================


def get_collection(service: VectorStoreService):
    """Return the underlying Chroma collection."""

    collection = getattr(
        service.store,
        "collection",
        None,
    )

    if collection is None:
        raise RuntimeError(
            "Could not access the underlying Chroma collection."
        )

    return collection


def test_collection(collection) -> int:
    """Inspect collection metadata and document count."""

    section("[4] CHROMA COLLECTION")

    count = collection.count()

    print(
        "Collection name    :",
        collection.name,
    )

    print(
        "Document count     :",
        count,
    )

    print(
        "Metadata           :",
        collection.metadata,
    )

    if count == 0:
        raise RuntimeError(
            "Chroma collection contains ZERO documents."
        )

    print("Status             : PASS")

    return count


# ============================================================================
# STORED DOCUMENT TEST
# ============================================================================


def test_stored_documents(
    collection,
    max_documents: int = 10,
) -> None:
    """Inspect the first stored documents."""

    section("[5] STORED DOCUMENT INSPECTION")

    result = collection.get(
        include=[
            "documents",
            "metadatas",
        ]
    )

    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    print(
        "IDs retrieved      :",
        len(ids),
    )

    print(
        "Documents          :",
        len(documents),
    )

    print(
        "Metadata records   :",
        len(metadatas),
    )

    if not documents:
        raise RuntimeError(
            "Collection contains IDs but no document text."
        )

    for index in range(
        min(
            max_documents,
            len(ids),
        )
    ):
        print()
        print("-" * 70)
        print(f"DOCUMENT {index + 1}")
        print("-" * 70)

        print("ID:")
        print(ids[index])

        metadata = (
            metadatas[index]
            if index < len(metadatas)
            else {}
        )

        document = (
            documents[index]
            if index < len(documents)
            else ""
        )

        print()
        print("Metadata:")
        print(metadata)

        print()
        print("Content:")
        print(truncate(document, 1000))


# ============================================================================
# EMBEDDING TEST
# ============================================================================


def test_embedding_service() -> tuple[
    EmbeddingService,
    list[float],
]:
    """Load embedding model and generate a query vector."""

    section("[6] EMBEDDING SERVICE")

    embedding_service = EmbeddingService()

    print(
        "Provider            :",
        embedding_service.provider.__class__.__name__,
    )

    query = (
        "Tell me about Bharati's education "
        "and work experience."
    )

    print()
    print("Query:")
    print(query)

    query_embedding = embedding_service.embed_query(
        query
    )

    dimension = safe_length(
        query_embedding
    )

    print()
    print(
        "Query embedding dim :",
        dimension,
    )

    if dimension == 0:
        raise RuntimeError(
            "Embedding model returned an empty vector."
        )

    print("Status              : PASS")

    return (
        embedding_service,
        query_embedding,
    )


# ============================================================================
# EMBEDDING HEALTH
# ============================================================================


def test_embedding_health(
    embedding_service: EmbeddingService,
) -> None:
    """Run the embedding provider health check."""

    section("[7] EMBEDDING HEALTH")

    healthy = embedding_service.health_check()

    print(
        "Health              :",
        healthy,
    )

    if not healthy:
        raise RuntimeError(
            "Embedding provider health check FAILED."
        )

    print("Status              : PASS")


# ============================================================================
# VECTOR SEARCH
# ============================================================================


def test_vector_search(
    service: VectorStoreService,
    query_embedding: list[float],
) -> dict:
    """Run semantic vector search against Chroma."""

    section("[8] SEMANTIC VECTOR SEARCH")

    k = 8

    print("Requested top_k    :", k)
    print(
        "Embedding dimension:",
        len(query_embedding),
    )

    search_result = service.search(
        embedding=query_embedding,
        k=k,
    )

    if not isinstance(
        search_result,
        dict,
    ):
        raise RuntimeError(
            "Vector store search did not return a dictionary."
        )

    result_ids = (
        search_result.get("ids")
        or [[]]
    )[0]

    result_documents = (
        search_result.get("documents")
        or [[]]
    )[0]

    result_metadatas = (
        search_result.get("metadatas")
        or [[]]
    )[0]

    result_distances = (
        search_result.get("distances")
        or [[]]
    )[0]

    print(
        "Results returned   :",
        len(result_ids),
    )

    if not result_ids:
        raise RuntimeError(
            "Vector search returned ZERO results."
        )

    for index, result_id in enumerate(
        result_ids
    ):
        print()
        print("-" * 70)
        print(f"RESULT {index + 1}")
        print("-" * 70)

        print("ID:")
        print(result_id)

        if index < len(result_distances):
            print()
            print("Distance:")
            print(result_distances[index])

        if index < len(result_metadatas):
            print()
            print("Metadata:")
            print(result_metadatas[index])

        if index < len(result_documents):
            print()
            print("Content:")
            print(
                truncate(
                    result_documents[index],
                    1200,
                )
            )

    print()
    print("Status              : PASS")

    return search_result


# ============================================================================
# SEARCH QUALITY SUMMARY
# ============================================================================


def summarize_search(
    search_result: dict,
) -> None:
    """Print a compact search-quality summary."""

    section("[9] SEARCH SUMMARY")

    ids = (
        search_result.get("ids")
        or [[]]
    )[0]

    documents = (
        search_result.get("documents")
        or [[]]
    )[0]

    distances = (
        search_result.get("distances")
        or [[]]
    )[0]

    print(
        "Result IDs         :",
        len(ids),
    )

    print(
        "Result documents   :",
        len(documents),
    )

    print(
        "Result distances   :",
        len(distances),
    )

    if distances:
        print()
        print("Distances:")

        for index, distance in enumerate(
            distances,
            start=1,
        ):
            print(
                f"  {index}. {distance}"
            )

        print()
        print(
            "Best distance      :",
            min(distances),
        )

        print(
            "Worst distance     :",
            max(distances),
        )


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    """Run the complete vector-store diagnostic."""

    section("VECTOR STORE DIAGNOSTIC TEST")

    print(
        "This test is READ-ONLY."
    )

    print(
        "It will NOT delete or modify your Chroma database."
    )

    try:
        # --------------------------------------------------------------
        # 1. Configuration
        # --------------------------------------------------------------

        test_configuration()

        # --------------------------------------------------------------
        # 2. Vector store
        # --------------------------------------------------------------

        service = initialize_vector_store()

        # --------------------------------------------------------------
        # 3. Health
        # --------------------------------------------------------------

        test_health(service)

        # --------------------------------------------------------------
        # 4. Collection
        # --------------------------------------------------------------

        collection = get_collection(service)

        test_collection(collection)

        # --------------------------------------------------------------
        # 5. Stored documents
        # --------------------------------------------------------------

        test_stored_documents(
            collection,
            max_documents=10,
        )

        # --------------------------------------------------------------
        # 6. Embeddings
        # --------------------------------------------------------------

        (
            embedding_service,
            query_embedding,
        ) = test_embedding_service()

        # --------------------------------------------------------------
        # 7. Embedding health
        # --------------------------------------------------------------

        test_embedding_health(
            embedding_service
        )

        # --------------------------------------------------------------
        # 8. Search
        # --------------------------------------------------------------

        search_result = test_vector_search(
            service,
            query_embedding,
        )

        # --------------------------------------------------------------
        # 9. Summary
        # --------------------------------------------------------------

        summarize_search(
            search_result
        )

        # --------------------------------------------------------------
        # Final
        # --------------------------------------------------------------

        section("VECTOR STORE DIAGNOSTIC: PASS")

        print(
            "ChromaDB initialized successfully."
        )

        print(
            "Stored documents are readable."
        )

        print(
            "Embedding generation works."
        )

        print(
            "Semantic vector search returned results."
        )

    except Exception as exc:

        section("VECTOR STORE DIAGNOSTIC: FAILED")

        print(
            "Error type        :",
            type(exc).__name__,
        )

        print(
            "Error             :",
            str(exc),
        )

        print()
        print(
            "The vector-store pipeline cannot be considered "
            "healthy until this error is resolved."
        )

        raise


# ============================================================================
# ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    main()