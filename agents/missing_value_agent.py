from agents.base_agent import BaseAgent
from services.file_loader import DATAFRAME_STORE


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
            missing_counts = df.isnull().sum()

            insights.append(
                f"Missing value analysis for {file_name}:"
            )

            for column, count in missing_counts.items():
                if count > 0:
                    insights.append(
                        f"Column '{column}' has {count} missing values."
                    )

            if missing_counts.sum() == 0:
                insights.append(
                    "No missing values detected."
                )

            insights.append(
                "Potential causes may include incomplete reporting, delayed project updates, or inconsistent data collection."
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