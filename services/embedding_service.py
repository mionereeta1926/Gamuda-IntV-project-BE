from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def generate_embedding(text):
    embedding = embedding_model.encode(text)

    return embedding.tolist()