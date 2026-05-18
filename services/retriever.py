import numpy as np

from services.embedding_service import generate_embedding
from services.logging_service import log_print
from services.vectorstore import load_vectorstore


def retrieve_documents(query, top_k=5, source_level=False):
    index, documents_store = load_vectorstore()

    query_embedding = generate_embedding(query)
    query_vector = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_vector, top_k)

    retrieved_docs = []

    for position, idx in enumerate(indices[0]):
        if idx < len(documents_store):
            doc = documents_store[idx].copy()
            doc["distance"] = float(distances[0][position])
            retrieved_docs.append(doc)

    if source_level and retrieved_docs:
        grouped = {}
        for doc in retrieved_docs:
            source = doc["metadata"].get("source", "Unknown")
            grouped.setdefault(source, []).append(doc)

        best_source = None
        best_score = None
        for source, docs in grouped.items():
            avg_distance = sum(d["distance"] for d in docs) / len(docs)
            if best_score is None or avg_distance < best_score:
                best_score = avg_distance
                best_source = source

        return sorted(grouped[best_source], key=lambda d: d["distance"]) if best_source else retrieved_docs
    log_print("Retrieved documents:", retrieved_docs)
    return retrieved_docs