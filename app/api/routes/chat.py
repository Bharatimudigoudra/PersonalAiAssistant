"""
Chat API routes.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.dependencies.llm import get_llm_service
from app.llm.services import LLMService
from app.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat")
def chat(
    request: ChatRequest,
    llm: LLMService = Depends(
        get_llm_service,
    ),
) -> ChatResponse:
    """
    Multi-turn conversation endpoint.
    """

    response = llm.generate(
        request.prompt,
    )

    history = llm.get_history()

    return ChatResponse(
        response=response,
        history_size=len(history),
    )

@router.post(
    "/chat/stream",
)
def stream_chat(
    request: ChatRequest,
    llm: LLMService = Depends(
        get_llm_service,
    ),
) -> StreamingResponse:
    """
    Stream the LLM response token by token.
    """

    return StreamingResponse(
        llm.stream(
            request.prompt,
        ),
        media_type="text/plain",
    )