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

    @staticmethod
    def _infer_document_metadata(
        file_path: str,
        sample_text: str,
    ) -> dict[str, str]:
        """
        Infer structured metadata from the file name and a sample of
        the content so retrieval can reason about document type and
        section without relying on fragile filename heuristics.
        """

        normalized_path = str(file_path or "").lower()
        text = (sample_text or "").lower()

        document_type = "general"
        section = "general"

        if "resume" in normalized_path or "cv" in normalized_path:
            document_type = "resume"
        elif "interview" in normalized_path:
            document_type = "interview"

        if "experience" in text or "work experience" in text:
            section = "experience"
        elif "introduce yourself" in text or "myself" in text:
            section = "intro"
        elif "education" in text:
            section = "education"
        elif "project" in text or "developed" in text:
            section = "projects"

        if document_type == "resume" and section == "general":
            if any(
                token in text
                for token in {
                    "work experience",
                    "experience",
                    "skill",
                    "project",
                    "education",
                }
            ):
                section = "experience"

        if document_type == "interview" and section == "general":
            if "introduce yourself" in text or "myself" in text:
                section = "intro"

        return {
            "document_type": document_type,
            "section": section,
        }

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

        document_metadata = self._infer_document_metadata(
            file_path,
            text,
        )

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
                    "document_type": document_metadata["document_type"],
                    "section": document_metadata["section"],
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