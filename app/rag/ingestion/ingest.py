"""
Run document ingestion.
"""

from app.rag.ingestion import DocumentIngestion


def main() -> None:
    pipeline = DocumentIngestion()

    pipeline.ingest(
        "data/documents/resume.pdf",
    )


if __name__ == "__main__":
    main()