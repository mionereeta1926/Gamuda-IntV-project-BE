from agents.base_agent import BaseAgent

from services.retriever import retrieve_documents
from services.llm_service import generate_chat_response


class DocumentQAAgent(BaseAgent):
    name = "Document Q&A Agent"

    def can_handle(self, query: str):
        keywords = [
            "report",
            "document",
            "risk",
            "timeline",
            "project",
            "summary",
        ]

        return any(word in query.lower() for word in keywords)

    def score(self, query: str) -> float:
        return 0.05 if query.strip() else 0.0

    def handle(self, query: str):
        docs = retrieve_documents(query, source_level=True)

        context = "\n\n".join([
            doc["content"] for doc in docs
        ])

        user_prompt = f"""
        Answer the question using ONLY the provided context.

        Do not repeat or quote the context in your answer.
        Do not repeat the question.
        Return only the answer text.

        Context:
        {context}

        Question:
        {query}
        """

        answer = generate_chat_response(
            system_prompt="You are a project intelligence assistant. Answer using only the context; do not include the context or question in your response.",
            user_prompt=user_prompt
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

        return {
            "agent": self.name,
            "answer": answer,
            "citations": citations,
            "_context": context,
        }