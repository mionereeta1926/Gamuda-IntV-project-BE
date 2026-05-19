from agents.base_agent import BaseAgent

from services.logging_service import log_print
from services.memory import build_chat_history
from services.retriever import retrieve_documents
from services.llm_service import generate_chat_response


class TabulateAgent(BaseAgent):
    name = "Tabulate Agent"

    def can_handle(self, query: str):
        keywords = [
            "Tabulate",
            "Show in Table",
            "Generate Table",
            "Generate Excel",
            "Generate CSV",
        ]

        return any(word in query.lower() for word in keywords)

    def score(self, query: str) -> float:
        keywords = [
            "Tabulate",
            "Show in Table",
            "Generate Table",
            "Generate Excel",
            "Generate CSV",
        ]
        matches = sum(1 for word in keywords if word in query.lower())
        return min(1.0, matches / len(keywords) + 0.1)

    def handle(self, memory_needed: bool, query: str, session_id: str | None = None):
        docs = retrieve_documents(query, source_level=False)

        context = "\n\n".join([
            doc["content"] for doc in docs
        ])

        if memory_needed:
            memory_history = build_chat_history(session_id) if session_id else []
        else:
            log_print("Memory not needed, skipping chat history.")
            memory_history = None

        user_prompt = f"""
        Answer the question using ONLY the provided context.

        Use the most recent conversation history if it helps answer the question.

        Context:
        {context}

        Question:
        {query}
        """

        answer = generate_chat_response(
            system_prompt="You are a project intelligence assistant mainly works for analysing project data. " \
            "The output must be in a proper table markdown format. No other format is allowed.",
            user_prompt=user_prompt,
            chat_history=memory_history if memory_history else None,
        )

        citations = []
        seen = set()

        for doc in docs:
            filename = doc["metadata"].get("source", "Unknown").split("/")[-1].split("\\")[-1]
            sheet = doc["metadata"].get("sheet")

            if sheet is not None:
                citation_key = (filename, sheet)
            else:
                citation_key = (filename, doc["metadata"].get("page", 1))

            if citation_key in seen:
                continue

            seen.add(citation_key)

            citation = {"source": filename}
            if sheet is not None:
                citation["sheet"] = sheet
            else:
                citation["page"] = doc["metadata"].get("page", 1)

            citations.append(citation)
        
        unique_citations= list(
            {item["source"]: item for item in citations}.values()
        )

        return {
            "agent": self.name,
            "answer": answer,
            "citations": unique_citations,
            "_context": context,
        }