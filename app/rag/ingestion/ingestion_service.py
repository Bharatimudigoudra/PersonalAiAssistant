"""
Document ingestion pipeline.
"""

from uuid import uuid4

from app.core.logging import logger
from app.embeddings.embedding_service import EmbeddingService
from app.rag.bm25.bm25_service import get_bm25_service
from app.rag.chunking.text_chunker import TextChunker
from app.rag.loaders.loader_factory import LoaderFactory
from app.models.retrieved_document import RetrievedDocument
from app.vectorstore.vectorstore_service import VectorStoreService


class DocumentIngestion:
    """
    Loads, chunks, embeds and stores documents
    in the retrieval system.
    """

    def __init__(self) -> None:

        self.loader_factory = LoaderFactory()
        self.chunker = TextChunker()
        self.embedding_service = EmbeddingService()
        self.vectorstore = VectorStoreService()
        self.bm25 = get_bm25_service()

        logger.info(
            "DocumentIngestion initialized."
        )

    def ingest(
        self,
        file_path: str,
    ) -> None:
        """
        Load, chunk, embed and index a document.
        """

        logger.info(
            "Starting ingestion: {}",
            file_path,
        )

        # --------------------------------------------------
        # Load document
        # --------------------------------------------------

        loader = self.loader_factory.create(
            file_path,
        )

        text = loader.load(
            file_path,
        )

        # --------------------------------------------------
        # Split into chunks
        # --------------------------------------------------

        chunks = self.chunker.split(
            text,
        )

        if not chunks:

            logger.warning(
                "No chunks generated from document."
            )

            return

        logger.info(
            "Generated {} chunks.",
            len(chunks),
        )

        # --------------------------------------------------
        # Generate embeddings
        # --------------------------------------------------

        embeddings = (
            self.embedding_service.embed_documents(
                chunks,
            )
        )

        if len(embeddings) != len(chunks):

            logger.error(
                "Embedding generation failed."
            )

            raise RuntimeError(
                "Embedding count does not match chunk count."
            )

        # --------------------------------------------------
        # Prepare metadata
        # --------------------------------------------------

        ids: list[str] = []
        metadatas: list[dict] = []

        for index in range(
            len(chunks),
        ):

            ids.append(
                str(uuid4())
            )

            metadatas.append(
                {
                    "source": file_path,
                    "chunk": index,
                }
            )

        # --------------------------------------------------
        # Store in vector database
        # --------------------------------------------------

        self.vectorstore.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            "Stored {} chunks in vector database.",
            len(chunks),
        )

        # --------------------------------------------------
        # Build BM25 index
        # --------------------------------------------------

        bm25_documents: list[
            RetrievedDocument
        ] = []

        for chunk, metadata in zip(
            chunks,
            metadatas,
        ):

            bm25_documents.append(
                RetrievedDocument(
                    content=chunk,
                    distance=0.0,
                    metadata=metadata,
                )
            )

        self.bm25.build_index(
            bm25_documents,
        )

        logger.info(
            "BM25 index updated ({} chunks).",
            len(bm25_documents),
        )

        logger.info(
            "Document ingestion completed successfully."
        )