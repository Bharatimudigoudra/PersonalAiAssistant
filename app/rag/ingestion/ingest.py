"""
Multi-document ingestion entry point.

Discovers all supported documents inside data/documents
and ingests them into the persistent retrieval system.
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
    """Ingest all supported documents."""

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {DOCUMENTS_DIR}"
        )

    files = sorted(
        path
        for path in DOCUMENTS_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    logger.info(
        f"Found {len(files)} document(s) for ingestion."
    )

    if not files:
        logger.warning("No supported documents found.")
        return

    pipeline = DocumentIngestion()

    successful = 0
    failed = 0

    print("\n" + "=" * 70)
    print("PERSONAL AI ASSISTANT - DOCUMENT INGESTION")
    print("=" * 70)

    for file_path in files:

        print(f"\nIngesting: {file_path.name}")

        try:
            pipeline.ingest(str(file_path))

            successful += 1

            print(
                f"[OK] Successfully ingested: "
                f"{file_path.name}"
            )

        except Exception:

            failed += 1

            logger.exception(
                f"Failed to ingest {file_path}"
            )

            print(
                f"[ERROR] Failed: {file_path.name}"
            )

    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    print(f"Total files : {len(files)}")
    print(f"Successful  : {successful}")
    print(f"Failed      : {failed}")
    print("=" * 70)


if __name__ == "__main__":
    main()