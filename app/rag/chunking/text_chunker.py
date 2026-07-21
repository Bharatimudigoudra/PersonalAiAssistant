"""
Text chunking utility.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import rag
from app.core.logging import logger


class TextChunker:
    """
    Splits documents into overlapping chunks.
    """

    def __init__(self) -> None:

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=rag.chunk_size,
            chunk_overlap=rag.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

        logger.info(
            "TextChunker initialized "
            "(chunk_size={}, overlap={})",
            rag.chunk_size,
            rag.chunk_overlap,
        )

    def split(
        self,
        text: str,
    ) -> list[str]:
        """
        Split text into chunks.
        """

        chunks = self.splitter.split_text(text)

        logger.info(
            "Created {} chunks.",
            len(chunks),
        )

        return chunks