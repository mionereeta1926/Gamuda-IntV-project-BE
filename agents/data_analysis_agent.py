import pandas as pd

from agents.base_agent import BaseAgent
from services.file_loader import DATAFRAME_STORE
from services.llm_service import generate_chat_response
from services.memory import build_chat_history
from services.retriever import retrieve_documents


class DataAnalysisAgent(BaseAgent):
    name = "Data Analysis Agent"

    def can_handle(self, query: str):
        keywords = [
            "budget",
            "cost",
            "average",
            "sum",
            "spreadsheet",
            "excel",
            "csv",
            "trend",
            "financial",
        ]

        return any(word in query.lower() for word in keywords)

    def score(self, query: str) -> float:
        keywords = [
            "budget",
            "cost",
            "average",
            "sum",
            "spreadsheet",
            "excel",
            "csv",
            "trend",
            "financial",
        ]
        matches = sum(1 for word in keywords if word in query.lower())
        return min(1.0, matches / len(keywords) + 0.1)

    def handle(self, query: str, session_id: str | None = None):
        if not DATAFRAME_STORE:
            return {
                "agent": self.name,
                "answer": "No spreadsheet data uploaded.",
                "citations": [],
            }

        insights = []
        citations = []

        for file_name, df in DATAFRAME_STORE.items():
            if isinstance(df, dict):
                for sheet_name, sheet_df in df.items():
                    # insights.append(
                    #     f"Dataset {file_name} (sheet: {sheet_name}) contains {sheet_df.shape[0]} rows and {sheet_df.shape[1]} columns."
                    # )
                    insights.append(
                        f"Columns available: {', '.join(sheet_df.columns)}"
                    )

                    numeric_columns = sheet_df.select_dtypes(include="number")
                    if not numeric_columns.empty:
                        means = numeric_columns.mean().to_dict()
                        insights.append(f"Numeric column averages: {means}")

                    filename = file_name.split("/")[-1].split("\\")[-1]
                    citations.append({
                        "source": filename,
                        "page": f"{sheet_name}",
                    })
            else:
                # insights.append(
                #     f"Dataset {file_name} contains {df.shape[0]} rows and {df.shape[1]} columns."
                # )
                insights.append(
                    f"Columns available: {', '.join(df.columns)}"
                )

                numeric_columns = df.select_dtypes(include="number")
                if not numeric_columns.empty:
                    means = numeric_columns.mean().to_dict()
                    insights.append(f"Numeric column averages: {means}")

                filename = file_name.split("/")[-1].split("\\")[-1]
                citations.append({
                    "source": filename,
                    "page": "sheet-data",
                })

        docs = retrieve_documents(query, source_level=True)

        context = "\n\n".join([
            doc["content"] for doc in docs
        ])

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
            system_prompt="You are a project intelligence assistant mainly works for analyzing spreadsheet data. Answer using only the context; do not include the context or question in your response.",
            user_prompt=user_prompt,
            chat_history=memory_history if memory_history else None,
        )

        insights.append("\n\n" + answer)

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
            "answer": "\n".join(insights),
            "citations": citations,
            "_context": context,
        }