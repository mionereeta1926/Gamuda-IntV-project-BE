from agents.document_agent import DocumentQAAgent
from agents.data_analysis_agent import DataAnalysisAgent
from agents.missing_value_agent import MissingValueAgent


class RouterAgent:
    def __init__(self):
        self.agents = [
            MissingValueAgent(),
            DataAnalysisAgent(),
            DocumentQAAgent(),
        ]

    def route(self, query):
        for agent in self.agents:
            if agent.can_handle(query):
                return agent.handle(query)

        return {
            "agent": "Fallback Agent",
            "answer": "No suitable agent found for the query.",
            "citations": [],
        }