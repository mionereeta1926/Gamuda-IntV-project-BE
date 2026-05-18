import numpy as np

from services.vectorstore import load_vectorstore
from services.embedding_service import generate_embedding


def retrieve_documents(query, top_k=5):
    index, documents_store = load_vectorstore()

    query_embedding = generate_embedding(query)

    query_vector = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_vector, top_k)

    retrieved_docs = []

    for idx in indices[0]:
        if idx < len(documents_store):
            retrieved_docs.append(documents_store[idx])

    return retrieved_docs