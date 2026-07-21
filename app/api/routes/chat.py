from fastapi import APIRouter, Depends

from app.dependencies.llm import get_llm_service
from app.llm.services import LLMService
from app.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    llm: LLMService = Depends(get_llm_service),
) -> ChatResponse:

    response = llm.generate(request.prompt)

    return ChatResponse(response=response)