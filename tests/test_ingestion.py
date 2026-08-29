from app.rag.ingestion.ingestion_service import DocumentIngestion
from app.vectorstore.vectorstore_service import VectorStoreService


def test_document_ingestion_persists_metadata_for_real_file() -> None:
    pipeline = DocumentIngestion()
    pipeline.ingest("data/documents/resume.pdf")

    vectorstore = VectorStoreService()
    collection = vectorstore.store.collection
    result = collection.get(include=["metadatas"])
    metadatas = result.get("metadatas") or []

    assert metadatas, "resume ingestion should store at least one chunk"
    assert any(
        metadata.get("document_type") == "resume"
        for metadata in metadatas
    ), "stored chunks should include resume document_type metadata"
    assert any(
        metadata.get("section") in {"experience", "projects", "education", "general"}
        for metadata in metadatas
    ), "stored chunks should include a section value"
