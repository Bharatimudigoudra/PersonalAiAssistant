"""
Text document loader.
"""

from pathlib import Path

from app.core.logging import logger
from app.rag.loaders.base_loader import BaseLoader


class TextLoader(BaseLoader):
    """
    Loads plain text and Markdown files.
    """

    def load(
        self,
        file_path: str,
    ) -> str:

        logger.info(
            "Loading text document: {}",
            file_path,
        )

        text = Path(file_path).read_text(
            encoding="utf-8"
        )

        logger.info(
            "Loaded {} characters.",
            len(text),
        )

        return text