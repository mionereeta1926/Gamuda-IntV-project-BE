import os

from google import genai
from google.genai.types import EmbedContentConfig

from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"


load_dotenv(dotenv_path=ENV_PATH)

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)


def generate_embedding(text):
    response = client.models.embed_content(
        model="text-embedding-005",
        contents=text,
        config=EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT"
        )
    )

    return response.embeddings[0].values