"""
Run document ingestion for all supported documents.

This script scans data/documents/ and ingests every supported file.
"""

from pathlib import Path

from app.core.logging import logger
from app.rag.ingestion import DocumentIngestion


DOCUMENTS_DIR = Path("data/documents")

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
}


def main() -> None:
    """Ingest all supported documents from the documents directory."""

    if not DOCUMENTS_DIR.exists():
        logger.error(
            "Documents directory does not exist: %s",
            DOCUMENTS_DIR,
        )
        return

    files = [
        file_path
        for file_path in DOCUMENTS_DIR.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        logger.warning(
            "No supported documents found in: %s",
            DOCUMENTS_DIR,
        )
        return

    logger.info(
        "Found %d document(s) for ingestion.",
        len(files),
    )

    pipeline = DocumentIngestion()

    successful = 0
    failed = 0

    for file_path in files:
        print(f"\n{'=' * 70}")
        print(f"Ingesting: {file_path.name}")
        print(f"{'=' * 70}")

        try:
            pipeline.ingest(str(file_path))

            successful += 1

            logger.info(
                "Successfully ingested: %s",
                file_path,
            )

        except Exception:
            failed += 1

            logger.exception(
                "Failed to ingest: %s",
                file_path,
            )

    print(f"\n{'=' * 70}")
    print("INGESTION SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total files : {len(files)}")
    print(f"Successful  : {successful}")
    print(f"Failed      : {failed}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()