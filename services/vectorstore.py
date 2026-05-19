import os
import faiss
import pickle
import numpy as np
from services.embedding_service import generate_embedding

DB_PATH = "data/faiss_index"

documents_store = []

index = None


def initialize_faiss():
    global index

    if index is None:
        sample_embedding = generate_embedding("test")
        dimension = len(sample_embedding)

        index = faiss.IndexFlatL2(dimension)


def save_to_vectorstore(chunks):
    global documents_store

    initialize_faiss()

    vectors = []

    for chunk in chunks:
        embedding = generate_embedding(chunk.page_content)

        vectors.append(embedding)

        documents_store.append({
            "content": chunk.page_content,
            "metadata": chunk.metadata
        })

    vectors = np.array(vectors).astype("float32")

    index.add(vectors)

    faiss.write_index(index, f"{DB_PATH}.index")

    with open(f"{DB_PATH}.pkl", "wb") as f:
        pickle.dump(documents_store, f)


def load_vectorstore():
    global index
    global documents_store

    index = faiss.read_index(f"{DB_PATH}.index")

    with open(f"{DB_PATH}.pkl", "rb") as f:
        documents_store = pickle.load(f)

    return index, documents_store