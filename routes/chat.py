from fastapi import APIRouter
from pydantic import BaseModel

from agents.router_agent import RouterAgent
from services.memory import save_memory

router = APIRouter()

router_agent = RouterAgent()


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


@router.post("/chat")
def chat(request: ChatRequest):
    response = router_agent.route(request.question)

    save_memory(
        request.session_id,
        request.question,
        response["answer"],
    )

    return response