SESSION_MEMORY = {}


def save_memory(session_id, query, answer):
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = []

    SESSION_MEMORY[session_id].append({
        "query": query,
        "answer": answer,
    })


def get_memory(session_id):
    return SESSION_MEMORY.get(session_id, [])