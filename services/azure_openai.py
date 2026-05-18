import os

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# =============================
# Azure OpenAI (Chat Model)
# =============================

openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
openai_subscription_key = os.environ["AZURE_OPENAI_KEY"]
chat_deployment = "gpt-5-chat"

# =============================
# Azure Embedding Model
# =============================

embedding_deployment = "text-embedding-3-large"
embedding_endpoint = os.environ["AZURE_EMBEDDING_ENDPOINT"]
embedding_subscription_key = os.environ["AZURE_EMBEDDING_KEY"]

# =============================
# Initialize Clients
# =============================

openai_client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=openai_endpoint,
    api_key=openai_subscription_key
)

vector_client = AzureOpenAI(
    api_version="2024-02-01",
    azure_endpoint=embedding_endpoint,
    api_key=embedding_subscription_key,
)

# =============================
# Chat Completion Function
# =============================

def generate_chat_response(system_prompt, user_prompt):
    print("Generating chat response ...")
    response = openai_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        max_tokens=1024,
        temperature=0.3,
        model=chat_deployment
    )

    return response.choices[0].message.content

# =============================
# Embedding Function
# =============================

def generate_embedding(text):
    print("Generating embedding ...")
    response = vector_client.embeddings.create(
        model=embedding_deployment,
        input=[text]
    )

    return response.data[0].embedding