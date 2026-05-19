SESSION_MEMORY = {}
MEMORY_HISTORY_LIMIT = 3


def save_memory(session_id, query, answer):
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = []

    SESSION_MEMORY[session_id].append({
        "query": query,
        "answer": answer,
    })


def get_memory(session_id):
    return SESSION_MEMORY.get(session_id, [])


def get_recent_memory(session_id, limit=MEMORY_HISTORY_LIMIT):
    return get_memory(session_id)[-limit:]


def build_chat_history(session_id, limit=MEMORY_HISTORY_LIMIT):
    history = []
    for entry in get_recent_memory(session_id, limit=limit):
        history.append({"role": "user", "content": entry["query"]})
        history.append({"role": "assistant", "content": entry["answer"]})
    return history