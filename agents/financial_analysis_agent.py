import pandas as pd

from agents.base_agent import BaseAgent
from services.file_loader import DATAFRAME_STORE
from services.llm_service import generate_chat_response
from services.logging_service import log_print
from services.memory import build_chat_history
from services.retriever import retrieve_documents


class FinancialAnalysisAgent(BaseAgent):
    name = "Financial Analysis Agent"

    def can_handle(self, query: str):
        keywords = [
            "cost",
            "average",
            "sum",
            "lost",
            "gain",
            "financial",
        ]

        return any(word in query.lower() for word in keywords)

    def score(self, query: str) -> float:
        keywords = [
            "cost",
            "average",
            "sum",
            "lost",
            "gain",
            "financial",
        ]
        matches = sum(1 for word in keywords if word in query.lower())
        return min(1.0, matches / len(keywords) + 0.1)

    def handle(self, memory_needed: bool, query: str, session_id: str | None = None):
        

        insights = []
        citations = []
        dataframe_answer = []

        if not DATAFRAME_STORE:
            insights.append("No spreadsheet data uploaded! Looking into other documents!\n\n") 
        else:
            for file_name, df in DATAFRAME_STORE.items():
                if isinstance(df, dict):
                    iterables = df.items()
                else:
                    iterables = [(None, df)]

                for sheet_name, sheet_df in iterables:

                    dataframe_answer.append(f"Columns available: {', '.join(sheet_df.columns)}")

                    numeric_df = sheet_df.select_dtypes(include=["number"])

                    if not numeric_df.empty:
                        col_stats = {}

                        for col in numeric_df.columns:
                            series = numeric_df[col].dropna()

                            col_stats[col] = {
                                "sum": float(series.sum()),
                                "mean": float(series.mean()),
                                "min": float(series.min()),
                                "max": float(series.max()),
                                "median": float(series.median()),
                                "std_dev": float(series.std()),
                                "variance": float(series.var()),
                                "count": int(series.count()),
                                "range": float(series.max() - series.min()) if len(series) > 0 else 0
                            }

                        dataframe_answer.append(f"Financial column summary: {col_stats}")

                    filename = file_name.split("/")[-1].split("\\")[-1]

                    citations.append({
                        "source": filename,
                        "page": sheet_name if sheet_name else "sheet-data",
                    })

        if not dataframe_answer:
            docs = retrieve_documents(query, source_level=False)

            context = "\n\n".join([
                doc["content"] for doc in docs
            ])
            context = "\n\n".join(dataframe_answer) + "\n\n" + context
        else:
            context = "\n\n".join(dataframe_answer)
            docs = []

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
            system_prompt="You are a project intelligence assistant mainly works for analyzing financial data. " \
            "Answer using only the context; do not include the full context or question in your response. Only include relevant data from it to support or show the financial reasoning/calculation." \
            "Don't halucinate the calculations, produce the financial answer in a table markdown format if possible.",
            user_prompt=user_prompt,
            chat_history=memory_history if memory_history else None,
        )

        insights.append("\n\n" + answer)

        seen = set()
        if docs:
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