import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from services.logging_service import log_print
import time

FALLBACK_RESPONSE = "I'm sorry, I couldn't generate a response right now. Please try again."
REQUEST_TIMEOUT_SECONDS = 30

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"


load_dotenv(dotenv_path=ENV_PATH)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found")

client = Groq(api_key=api_key)

# MODEL_NAME = "llama-3.3-70b-versatile"
MODEL_NAME = "openai/gpt-oss-120b"

basic_system_prompt = "Answer using the context; " \
    "But if asked for suggestions or insights, you can provide your reasoning and insights logically. " \
    "Always answer in a concise manner. You are an intelligent assistant that helps user in understanding their projects and prepare for projects "

basic_user_prompt = "If the query needs logical thinking and doesn't rely on the provided context, you can provide insights and suggestions based on your understanding and reasoning and ignore the context. " \
    "But if the query is asking for specific information, you should only answer based on the provided context. " \
    "Refer to the history if the query is related to previous messages. If the history is irrelevant, ignore it. " \
    "If the question is not medical campaign or project related or totally out of scope, you should politely decline to answer. "

def get_token_count_for_response():
    total_token = basic_system_prompt +" " + basic_user_prompt
    return total_token

# ----------------------------
# Response validation
# ----------------------------
def validate_llm_response(response):
    if (
        not response
        or not response.choices
        or not response.choices[0].message.content
    ):
        log_print("Empty LLM response received")
        return False, FALLBACK_RESPONSE

    return True, response.choices[0].message.content


# ----------------------------
# Error handler
# ----------------------------
def handle_llm_error(error):
    log_print(f"LLM failure: {str(error)}")
    return FALLBACK_RESPONSE


# ----------------------------
# Main function
# ----------------------------
def generate_chat_response(system_prompt, user_prompt, chat_history=None, intent_search=False):
    log_print("------USING GROQ TO ANSWER-----")

    if chat_history:
        log_print("Chat history:", len(chat_history))

    if chat_history is None:
        chat_history = []

    # ----------------------------
    # Prompt preparation
    # ----------------------------
    if not intent_search:
        system_prompt = basic_system_prompt + "\n\n" + system_prompt
        user_prompt = basic_user_prompt + "\n\n" + user_prompt

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_prompt})

    # ----------------------------
    # LLM call
    # ----------------------------
    try:
        start_time = time.time()

        if not intent_search:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.5,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        else:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.5,
                response_format={"type": "json_object"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

        log_print(f"LLM response time: {time.time() - start_time:.2f}s")

        # ----------------------------
        # Response validation
        # ----------------------------
        is_valid, result = validate_llm_response(response)
        return result

    except Exception as e:
        return handle_llm_error(e)