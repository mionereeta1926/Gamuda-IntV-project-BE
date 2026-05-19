import uuid

import numpy as np

from agents.document_agent import DocumentQAAgent
from agents.financial_analysis_agent import FinancialAnalysisAgent
from agents.tabulate_agent import TabulateAgent
from agents.missing_value_agent import MissingValueAgent
from services.embedding_service import generate_embedding
from services.logging_service import log_print


class RouterAgent:
    INTENT_TEXTS = {
        MissingValueAgent: [
            "missing values in data",
            "null values",
            "empty cells",
            "incomplete spreadsheet data",
            "dataset missing values analysis",
        ],
        TabulateAgent: [
            "Output in table format",
            "Tabulate the data",
            "Tabulate the findings",
            "Show in Table",
            "Generate Table",
            "Generate Spreadsheet",
            "Generate Excel",
            "Generate CSV",
            "show results in a table",
            "format as table",
            "present in tabular format",
            "convert findings into a table",
            "generate markdown table",
            "display rows and columns",
            "create structured table",
        ],
        FinancialAnalysisAgent: [
            "budget analysis",
            "cost analysis",
            "financial summary",
            "gain report",
            "loss report",
            "financial trends",
        ],
        DocumentQAAgent: [
            "project report",
            "document question",
            "risk assessment",
            "timeline overview",
            "project summary",
            "document-based question",
            "data trends",
        ],
    }

    def __init__(self):
        self.agents = [
            MissingValueAgent(),
            TabulateAgent(),
            DocumentQAAgent(),
            FinancialAnalysisAgent(),
        ]
        self.intent_embeddings = self._build_intent_embeddings()

    def _build_intent_embeddings(self):
        embeddings = {}

        for agent in self.agents:
            texts = self.INTENT_TEXTS.get(agent.__class__, [])
            embeddings[agent] = [
                np.array(generate_embedding(text), dtype=np.float32)
                for text in texts
            ]

        return embeddings

    def _cosine_similarity(self, a, b):
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    def _agent_score(self, agent, query_embedding):
        intent_vectors = self.intent_embeddings.get(agent, [])
        if not intent_vectors:
            return 0.0

        return max(self._cosine_similarity(query_embedding, intent) for intent in intent_vectors)

    def route(self, query, session_id: str = "default"):
        request_id = str(uuid.uuid4())
        query_embedding = np.array(generate_embedding(query), dtype=np.float32)
        scored_agents = [(agent, self._agent_score(agent, query_embedding)) for agent in self.agents]
        log_print(f"RouterAgent scored agents: {[(agent.name, score) for agent, score in scored_agents]}")
        best_agent, best_score = max(scored_agents, key=lambda item: item[1])

        document_agent = next(
            (agent for agent in self.agents if isinstance(agent, DocumentQAAgent)),
            None,
        )

        if best_score < 0.1 and document_agent is not None:
            return document_agent.handle_with_logging(query, request_id=request_id, session_id=session_id)

        return best_agent.handle_with_logging(query, request_id=request_id, session_id=session_id)