from app.models.retrieved_document import RetrievedDocument
from app.rag.ingestion.ingestion_service import DocumentIngestion
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


def test_document_retriever_prefers_resume_for_experience_questions() -> None:
    retriever = DocumentRetriever()

    intro = RetrievedDocument(
        content="Introduce yourself? Hi good afternoon. Myself Bharati mudi.",
        metadata={"source": "data/documents/interview.pdf"},
    )
    experience = RetrievedDocument(
        content="WORK EXPERIENCE AI Associate ResourcePro. Developed AI-driven automation workflows and performed data extraction and validation.",
        metadata={"source": "data/documents/resume.pdf"},
    )

    ranked = retriever._boost_by_source_relevance(
        "Describe your internship experience",
        [intro, experience],
    )

    assert ranked[0].metadata.get("source") == "data/documents/resume.pdf"


def test_document_ingestion_sets_explicit_document_metadata() -> None:
    resume_metadata = DocumentIngestion._infer_document_metadata(
        "data/documents/resume.pdf",
        "WORK EXPERIENCE AI Associate ResourcePro. Developed AI-driven automation workflows.",
    )
    interview_metadata = DocumentIngestion._infer_document_metadata(
        "data/documents/interview.pdf",
        "Introduce yourself? Hi good afternoon. Myself Bharati mudi.",
    )

    assert resume_metadata["document_type"] == "resume"
    assert resume_metadata["section"] == "experience"
    assert interview_metadata["document_type"] == "interview"
    assert interview_metadata["section"] == "intro"


def test_interview_assistant_uses_retrieved_context_in_prompt() -> None:
    from app.assistant.interview_assistant import InterviewAssistant

    class FakeLLM:
        def __init__(self) -> None:
            self.prompt = ""

        def generate_rag(self, prompt: str) -> str:
            self.prompt = prompt
            return "I worked at ResourcePro on AI automation workflows."

    class FakePromptBuilder:
        def build(self, question: str, documents):
            return (
                "SYSTEM",
                f"QUESTION={question}\nDOCS={len(documents)}\nFIRST={documents[0].content}",
            )

    class FakeRetriever:
        def retrieve(self, question: str):
            return [
                RetrievedDocument(
                    content="WORK EXPERIENCE AI Associate ResourcePro. Developed AI-driven automation workflows.",
                    metadata={"source": "data/documents/resume.pdf", "document_type": "resume"},
                )
            ]

    fake_llm = FakeLLM()
    fake_builder = FakePromptBuilder()
    fake_retriever = FakeRetriever()

    assistant = InterviewAssistant.__new__(InterviewAssistant)
    assistant.retriever = fake_retriever
    assistant.prompt_builder = fake_builder
    assistant.llm = fake_llm

    answer = assistant.answer_question("Describe your internship experience")

    assert answer.startswith("I worked")
    assert "ResourcePro" in fake_llm.prompt
    assert "DOCS=1" in fake_llm.prompt
