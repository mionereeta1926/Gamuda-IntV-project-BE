import numpy as np

from services.embedding_service import generate_embedding
from services.logging_service import log_print
from services.vectorstore import load_vectorstore


def retrieve_documents(
    query,
    top_k=20,
    source_level=False,
    max_distance=2.0,
    distance_window=1.0
):
    index, documents_store = load_vectorstore()
    doc_store_length = len(documents_store)
    log_print(f"Vectorstore loaded with {doc_store_length} documents")
    if top_k > doc_store_length:
        top_k = doc_store_length


    query_embedding = generate_embedding(query)
    query_vector = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_vector, top_k)

    retrieved_docs = []

    # First pass: collect valid docs under max_distance
    for position, idx in enumerate(indices[0]):

        if idx >= len(documents_store) or idx < 0:
            continue

        score = float(distances[0][position])

        log_print(f"Doc idx: {idx}, Score: {score} : {documents_store[idx]['content'][:100]}")

        # FILTER LEVEL 1:
        # Discard if distance too high
        if score > max_distance:
            continue

        doc = documents_store[idx].copy()
        doc["distance"] = score

        retrieved_docs.append(doc)

    # No valid docs
    if not retrieved_docs:
        log_print("No documents passed max distance filter")
        return []

    # Get best (lowest) distance
    best_distance = min(doc["distance"] for doc in retrieved_docs)

    # FILTER LEVEL 2:
    # Keep only docs close to best result
    filtered_docs = []

    for doc in retrieved_docs:

        # Example:
        # best = 1.1
        # allowed max = 2.1
        if doc["distance"] <= best_distance + distance_window:
            filtered_docs.append(doc)

    if not filtered_docs:
        log_print("No documents passed relative distance filter")
        return []
    else:
        log_print(f"{len(filtered_docs)} documents passed relative distance filter (window: {distance_window})")

    # Optional source grouping
    if source_level:
        grouped = {}

        for doc in filtered_docs:
            source = doc["metadata"].get("source", "Unknown")
            grouped.setdefault(source, []).append(doc)

        best_source = None
        best_avg_distance = None

        for source, docs in grouped.items():

            avg_distance = (
                sum(d["distance"] for d in docs) / len(docs)
            )

            if (
                best_avg_distance is None
                or avg_distance < best_avg_distance
            ):
                best_avg_distance = avg_distance
                best_source = source

        return sorted(
            grouped[best_source],
            key=lambda d: d["distance"]
        )

    return sorted(
        filtered_docs,
        key=lambda d: d["distance"]
    )