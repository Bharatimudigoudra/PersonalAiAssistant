"""
Vector Store Diagnostic Test.

Checks:
1. ChromaDB connection
2. Collection metadata
3. Number of stored chunks
4. Stored document content
5. Embedding dimensions
6. Semantic search
"""

from app.core.config import vectorstore
from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.vectorstore_service import VectorStoreService


def main() -> None:
    print("=" * 80)
    print("VECTOR STORE DIAGNOSTIC TEST")
    print("=" * 80)

    # ---------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------

    print("\n[1] Configuration")
    print("-" * 80)

    print("Provider          :", vectorstore.provider)
    print("Persist directory :", vectorstore.persist_directory)
    print("Collection        :", vectorstore.collection_name)

    # ---------------------------------------------------------------
    # Vector store
    # ---------------------------------------------------------------

    print("\n[2] Initializing vector store")
    print("-" * 80)

    service = VectorStoreService()
    store = service.store

    print("Store             :", store.__class__.__name__)

    # ---------------------------------------------------------------
    # Health
    # ---------------------------------------------------------------

    print("\n[3] Health check")
    print("-" * 80)

    healthy = service.health_check()

    print("Health            :", healthy)

    if not healthy:
        raise RuntimeError("Vector store health check failed.")

    # ---------------------------------------------------------------
    # Chroma information
    # ---------------------------------------------------------------

    print("\n[4] Collection information")
    print("-" * 80)

    collection = getattr(store, "collection", None)

    if collection is None:
        raise RuntimeError(
            "Underlying Chroma collection is not accessible."
        )

    count = collection.count()

    print("Document count    :", count)
    print("Collection name   :", collection.name)
    print("Metadata          :", collection.metadata)

    if count == 0:
        print("\nWARNING: Collection contains ZERO documents.")
        print("The ingestion pipeline must be checked next.")
        return

    # ---------------------------------------------------------------
    # Inspect stored documents
    # ---------------------------------------------------------------

    print("\n[5] Stored document inspection")
    print("-" * 80)

    result = collection.get(
        include=[
            "documents",
            "metadatas",
        ]
    )

    ids = result.get("ids", [])
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])

    print("IDs retrieved     :", len(ids))
    print("Documents         :", len(documents))
    print("Metadata records  :", len(metadatas))

    for index in range(min(10, len(ids))):

        print("\n" + "-" * 70)
        print(f"DOCUMENT {index + 1}")
        print("-" * 70)

        print("ID:")
        print(ids[index])

        print("\nMetadata:")
        print(metadatas[index])

        print("\nContent:")
        print(str(documents[index])[:1000])

    # ---------------------------------------------------------------
    # Embedding service
    # ---------------------------------------------------------------

    print("\n[6] Embedding service")
    print("-" * 80)

    embedding_service = EmbeddingService()

    query = "Tell me about Bharati's education and work experience."

    query_embedding = embedding_service.embed_query(query)

    print("Query             :", query)
    print("Embedding length  :", len(query_embedding))

    if not query_embedding:
        raise RuntimeError("Query embedding is empty.")

    # ---------------------------------------------------------------
    # Semantic search
    # ---------------------------------------------------------------

    print("\n[7] Semantic vector search")
    print("-" * 80)

    search_result = service.search(
        embedding=query_embedding,
        k=8,
    )

    result_ids = search_result.get("ids", [[]])[0]
    result_docs = search_result.get("documents", [[]])[0]
    result_distances = search_result.get(
        "distances",
        [[]],
    )[0]

    print("Results returned  :", len(result_ids))

    for index, result_id in enumerate(result_ids):

        print("\n" + "-" * 70)
        print(f"RESULT {index + 1}")
        print("-" * 70)

        print("ID:")
        print(result_id)

        if index < len(result_distances):
            print("Distance:")
            print(result_distances[index])

        if index < len(result_docs):
            print("Content:")
            print(str(result_docs[index])[:1000])

    # ---------------------------------------------------------------
    # Final result
    # ---------------------------------------------------------------

    print("\n" + "=" * 80)
    print("VECTOR STORE TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()