from agents.base_agent import BaseAgent
from services.llm_service import generate_chat_response
from services.file_loader import DATAFRAME_STORE
from services.memory import build_chat_history
from services.retriever import retrieve_documents


class MissingValueAgent(BaseAgent):
    name = "Missing Value Analysis Agent"

    def can_handle(self, query: str):
        keywords = [
            "missing",
            "null",
            "empty",
            "nan",
            "incomplete",
        ]

        return any(word in query.lower() for word in keywords)

    def score(self, query: str) -> float:
        keywords = [
            "missing",
            "null",
            "empty",
            "nan",
            "incomplete",
        ]
        matches = sum(1 for word in keywords if word in query.lower())
        return min(1.0, matches / len(keywords) + 0.1)

    def handle(self, query: str, session_id: str | None = None):
        insights = []
        citations = []
        dataframe_answer = []

        if not DATAFRAME_STORE:
            insights.append("No spreadsheet data uploaded! Looking into other documents!\n\n")  
        else:
            for file_name, df in DATAFRAME_STORE.items():
                if isinstance(df, dict):
                    # multiple sheets
                    for sheet_name, sheet_df in df.items():
                        missing_counts = sheet_df.isna().sum()
                        total_missing = int(missing_counts.sum())

                        if total_missing != 0:
                            for column, count in missing_counts.items():
                                if count > 0:
                                    dataframe_answer.append(f"Column '{column}' has {int(count)} missing values.")

                        filename = file_name.split("/")[-1].split("\\")[-1]
                        citations.append({"source": filename, "page": f"{sheet_name}"})
                else:
                    # single DataFrame (CSV or single-sheet)
                    missing_counts = df.isna().sum()
                    total_missing = int(missing_counts.sum())

                    if total_missing != 0:
                        for column, count in missing_counts.items():
                            if count > 0:
                                dataframe_answer.append(f"Column '{column}' has {int(count)} missing values.")

                    filename = file_name.split("/")[-1].split("\\")[-1]
                    citations.append({"source": filename, "page": "sheet-data"})

        docs = retrieve_documents(query, source_level=False)

        context = "\n\n".join([
            doc["content"] for doc in docs
        ])
        context = "\n\n".join(dataframe_answer) + "\n\n" + context

        memory_history = build_chat_history(session_id) if session_id else []

        user_prompt = f"""
        Answer the question using ONLY the provided context.

        Use the most recent conversation history if it helps answer the question.

        Context:
        {context}

        Question:
        {query}
        """

        answer = generate_chat_response(
            system_prompt="You are a project intelligence assistant mainly works for analysing missing or unknown values in spreadsheet data." \
            "Your task is to identify, reason and provide insights about missing or unknown values.",
            user_prompt=user_prompt,
            chat_history=memory_history if memory_history else None,
        )

        insights.append(answer)

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
            "answer": "\n".join(insights),
            "citations": unique_citations,
            "_context": context,
        }