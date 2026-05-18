import pandas as pd

from agents.base_agent import BaseAgent
from services.file_loader import DATAFRAME_STORE


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

    def handle(self, query: str):
        if not DATAFRAME_STORE:
            return {
                "agent": self.name,
                "answer": "No spreadsheet data uploaded.",
                "citations": [],
            }

        insights = []
        citations = []

        for file_name, df in DATAFRAME_STORE.items():
            insights.append(
                f"Dataset {file_name} contains {df.shape[0]} rows and {df.shape[1]} columns."
            )

            insights.append(
                f"Columns available: {', '.join(df.columns)}"
            )

            numeric_columns = df.select_dtypes(include="number")

            if not numeric_columns.empty:
                means = numeric_columns.mean().to_dict()

                insights.append(
                    f"Numeric column averages: {means}"
                )

            filename = file_name.split("/")[-1].split("\\")[-1]
            citations.append({
                "source": filename,
                "page": "sheet-data",
            })

        return {
            "agent": self.name,
            "answer": "\n".join(insights),
            "citations": citations,
        }