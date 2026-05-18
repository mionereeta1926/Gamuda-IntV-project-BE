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
            if isinstance(df, dict):
                # multiple sheets
                for sheet_name, sheet_df in df.items():
                    missing_counts = sheet_df.isna().sum()
                    total_missing = int(missing_counts.sum())

                    if total_missing != 0:
                    #     insights.append(f"No missing values detected in {file_name} (sheet: {sheet_name}).")
                    # else:
                        for column, count in missing_counts.items():
                            if count > 0:
                                insights.append(f"Column '{column}' has {int(count)} missing values.")

                    filename = file_name.split("/")[-1].split("\\")[-1]
                    citations.append({"source": filename, "page": f"{sheet_name}"})
            else:
                # single DataFrame (CSV or single-sheet)
                missing_counts = df.isna().sum()
                total_missing = int(missing_counts.sum())

                if total_missing != 0:
                #     insights.append(f"No missing values detected in {file_name}.")
                # else:
                    for column, count in missing_counts.items():
                        if count > 0:
                            insights.append(f"Column '{column}' has {int(count)} missing values.")

                filename = file_name.split("/")[-1].split("\\")[-1]
                citations.append({"source": filename, "page": "sheet-data"})

        return {
            "agent": self.name,
            "answer": "\n".join(insights),
            "citations": citations,
        }