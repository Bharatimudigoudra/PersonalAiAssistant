from fastapi import APIRouter

from app.rag.rag_service import RAGService
from app.schemas import RAGRequest, RAGResponse

router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
)

rag = RAGService()


@router.post(
    "/chat",
    response_model=RAGResponse,
)
def rag_chat(request: RAGRequest):

    answer = rag.ask(request.question)

    return RAGResponse(
        answer=answer,
    )