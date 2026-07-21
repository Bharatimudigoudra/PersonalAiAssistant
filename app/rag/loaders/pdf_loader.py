"""
PDF document loader.
"""

import fitz

from app.core.logging import logger
from app.rag.loaders.base_loader import BaseLoader


class PDFLoader(BaseLoader):
    """
    Loads PDF documents.
    """

    def load(
        self,
        file_path: str,
    ) -> str:

        logger.info(
            "Loading PDF: {}",
            file_path,
        )

        document = fitz.open(file_path)

        pages = []

        for page in document:
            pages.append(page.get_text())

        document.close()

        text = "\n".join(pages)

        logger.info(
            "Loaded {} characters.",
            len(text),
        )

        return text