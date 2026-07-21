from pathlib import Path

from app.rag.ingestion import DocumentIngestion

pipeline = DocumentIngestion()

file_path = Path("data/documents/resume.pdf")

pipeline.ingest(str(file_path))

print("Ingestion completed successfully!")