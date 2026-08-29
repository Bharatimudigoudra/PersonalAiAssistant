from app.models.retrieved_document import RetrievedDocument
from app.rag.retrieval import DocumentRetriever


def test_document_retriever_defers_heavy_initialization() -> None:
    retriever = DocumentRetriever()

    assert retriever.embedding_service is None
    assert retriever.vectorstore is None
    assert retriever.reranker is None


def test_document_retriever_returns_relevant_interview_context() -> None:
    retriever = DocumentRetriever()

    docs = retriever.retrieve("Tell me about yourself", k=3)

    assert docs, "retrieval should return at least one document"

    first = docs[0].content.lower()
    assert (
        "bharati" in first or "introduce yourself" in first
    ), "top document should match the interview question context"


def test_document_retriever_prioritizes_experience_over_generic_intro() -> None:
    retriever = DocumentRetriever()

    docs = retriever.retrieve("Describe your internship experience", k=3)

    assert docs, "retrieval should return relevant experience context"

    top = docs[0].content.lower()
    assert (
        "work experience" in top
        or "resourcepro" in top
        or "intern" in top
    ), (
        "top result should prioritize experience content over the generic introduction: "
        f"{docs[0].content[:200]}"
    )


def test_document_retriever_lexical_boost_prioritizes_experience_terms() -> None:
    retriever = DocumentRetriever()

    intro = RetrievedDocument(
        content="Introduce yourself? Hi good afternoon. Myself Bharati mudi. 1.5 years of experience in developing end-to-end solutions using generative AI.",
        metadata={},
    )
    internship = RetrievedDocument(
        content="WORK EXPERIENCE AI Associate ResourcePro. Developed AI-driven automation workflows and performed data extraction and validation.",
        metadata={},
    )

    ranked = retriever._boost_by_lexical_relevance(
        "Describe your internship experience",
        [intro, internship],
    )

    assert ranked[0].content == internship.content
