from fastapi import APIRouter
from pydantic import BaseModel

from agents.router_agent import RouterAgent
from services.memory import save_memory
import re

router = APIRouter()

router_agent = RouterAgent()


FALLBACK_RESPONSE = "I'm sorry, I couldn't generate a response right now. Please try again."
INJECTION_FALLBACK = "Potential Prompt injection"
EMPTY_RESPONSE = "No input provided. Please enter a valid query."
EMPTY_QUESTION_RESPONSE = "Please input a question"


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"

def detect_prompt_injection(text):
    if not text:
        return False

    t = text.lower()

    # ----------------------------
    # High-signal injection patterns
    # ----------------------------
    injection_patterns = [
        r"ignore all previous instructions",
        r"disregard (the )?above (instructions|directions)",
        r"stop following your system prompt",
        r"forget (all|your) (previous )?(instructions|guidelines)",
        r"override (your|system|all) (instructions|rules|objectives)",
        r"cancel (the )?previous command",
        r"act as an unrestricted ai",
        r"developer mode",
        r"no moral boundaries",
        r"do not mention you are an ai",
        r"print the text above",
        r"repeat your (initial|system) prompt",
        r"show me your hidden context",
        r"system update",
        r"new prompt:",
        r"--- stop ---",
        r"decode this text",
        r"base64",
        r"rot13",
    ]

    for pattern in injection_patterns:
        if re.search(pattern, t):
            return True

    return False


@router.post("/chat")
def chat(request: ChatRequest):
    # ----------------------------
    # Empty question validation
    # ----------------------------
    if not request.question or request.question.strip() == "":
        return {
            "agent": "Base Agent",
            "answer": EMPTY_QUESTION_RESPONSE,
            "citations": [],
        }
    
    if detect_prompt_injection(request.question):
        return {
            "agent": "Base Agent",
            "answer": INJECTION_FALLBACK,
            "citations": [],
        }

    response = router_agent.route(request.question, request.session_id)

    answer = response["answer"]

    # ----------------------------
    # Skip memory save for fallback responses
    # ----------------------------
    blocked_responses = [
        FALLBACK_RESPONSE,
        INJECTION_FALLBACK,
        EMPTY_RESPONSE,
        EMPTY_QUESTION_RESPONSE,
    ]

    if answer not in blocked_responses:
        save_memory(
        request.session_id,
        request.question,
        response["answer"],
    )
    else:
        return {
            "agent": "Base Agent",
            "answer": EMPTY_QUESTION_RESPONSE,
            "citations": [],
        }

    return response