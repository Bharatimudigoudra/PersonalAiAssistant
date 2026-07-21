"""
Document ingestion pipeline.
"""

from uuid import uuid4

from app.core.logging import logger
from app.embeddings.embedding_service import EmbeddingService
from app.rag.chunking.text_chunker import TextChunker
from app.rag.loaders.loader_factory import LoaderFactory
from app.vectorstore.vectorstore_service import VectorStoreService


class DocumentIngestion:
    """
    Ingests documents into the vector database.
    """

    def __init__(self) -> None:
        self.loader_factory = LoaderFactory()
        self.chunker = TextChunker()
        self.embedding_service = EmbeddingService()
        self.vectorstore = VectorStoreService()

    def ingest(self, file_path: str) -> None:
        """
        Load, chunk, embed and store a document.
        """

        logger.info("Starting ingestion: {}", file_path)

        # Load document
        loader = self.loader_factory.create(file_path)
        text = loader.load(file_path)

        # Split into chunks
        chunks = self.chunker.split(text)

        if not chunks:
            logger.warning("No chunks generated from document.")
            return

        # Generate embeddings for all chunks
        embeddings = self.embedding_service.embed_documents(chunks)

        ids = []
        metadatas = []

        for index in range(len(chunks)):
            ids.append(str(uuid4()))
            metadatas.append(
                {
                    "source": file_path,
                    "chunk": index,
                }
            )

        # Store in vector database
        self.vectorstore.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            "Document successfully ingested ({} chunks).",
            len(chunks),
        )