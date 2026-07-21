"""
Document loader factory.
"""

from pathlib import Path

from app.rag.loaders.base_loader import BaseLoader
from app.rag.loaders.pdf_loader import PDFLoader
from app.rag.loaders.text_loader import TextLoader


class LoaderFactory:
    """
    Creates document loaders.
    """

    @staticmethod
    def create(file_path: str) -> BaseLoader:

        extension = Path(file_path).suffix.lower()

        if extension == ".pdf":
            return PDFLoader()

        if extension in [".txt", ".md"]:
            return TextLoader()

        raise ValueError(
            f"Unsupported file type: {extension}"
        )