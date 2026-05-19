import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from services.logging_service import log_print

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"


load_dotenv(dotenv_path=ENV_PATH)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found")

client = Groq(api_key=api_key)

# MODEL_NAME = "llama-3.3-70b-versatile"
MODEL_NAME = "openai/gpt-oss-120b"


def generate_chat_response(system_prompt, user_prompt, chat_history=None):
    log_print("------USING GROQ TO ANSWER-----")
    if chat_history:
        log_print("Chat history:", len(chat_history))
    log_print("System prompt:", user_prompt)
    basic_system_prompt = "Answer using the context; " \
    "But if asked for suggestions or insights, you can provide your reasoning and insights logically. " \
    "Always answer in a concise manner. You are an intelligent assistant that helps user in understanding their projects and prepare for projects "
    system_prompt = basic_system_prompt + "\n\n" + system_prompt

    basic_user_prompt = "If the query needs logical thinking and doesn't rely on the provided context, you can provide insights and suggestions based on your understanding and reasoning and ignore the context. " \
    "But if the query is asking for specific information, you should only answer based on the provided context. " \
    "Refer to the history if the query is related to previous messages. If the history is irrelevant, ignore it. " \
    "If the question is not medical campaign or project related or totally out of scope, you should politely decline to answer. "
    user_prompt = basic_user_prompt + "\n\n" + user_prompt

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    if chat_history:
        messages.extend(chat_history)

    messages.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.5,
        # max_tokens=1024,
    )

    return response.choices[0].message.content