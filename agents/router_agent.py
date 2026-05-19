import uuid

import numpy as np
import json
from agents.document_agent import DocumentQAAgent
from agents.financial_analysis_agent import FinancialAnalysisAgent
from agents.tabulate_agent import TabulateAgent
from agents.missing_value_agent import MissingValueAgent
from services.embedding_service import generate_embedding
from services.llm_service import generate_chat_response
from services.logging_service import log_print
from services.memory import build_chat_history


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

    # def route(self, query, session_id: str = "default"):
    #     request_id = str(uuid.uuid4())
    #     query_embedding = np.array(generate_embedding(query), dtype=np.float32)
    #     scored_agents = [(agent, self._agent_score(agent, query_embedding)) for agent in self.agents]
    #     log_print(f"RouterAgent scored agents: {[(agent.name, score) for agent, score in scored_agents]}")
    #     best_agent, best_score = max(scored_agents, key=lambda item: item[1])

    #     document_agent = next(
    #         (agent for agent in self.agents if isinstance(agent, DocumentQAAgent)),
    #         None,
    #     )

    #     if best_score < 0.1 and document_agent is not None:
    #         return document_agent.handle_with_logging(query, request_id=request_id, session_id=session_id)

    #     return best_agent.handle_with_logging(query, request_id=request_id, session_id=session_id)
    def route(self, query, session_id: str = "default"):
        request_id = str(uuid.uuid4())
        log_print(f"--------RouterAgent received query------")
        memory_history = (
            build_chat_history(session_id) if session_id else []
        )
        log_print(f"--------RouterAgent retrived memory-------")

        available_agents = [agent.name for agent in self.agents]
        log_print(f"RouterAgent getting available agents: {available_agents}")

        system_prompt = f"""
You are an intelligent routing system for a multi-agent AI platform.

Your job is to:
1. Analyze the user's current query.
2. Analyze the conversation history.
3. Determine whether memory/history context is required.
4. Rewrite the user query into a cleaner and more machine-understandable prompt. 
Don't change the meaning, just clarify and structure it better for the agents. 
Don't add any information that isn't in the original query, just rephrase it for clarity and specificity.
5. Select the BEST agent from the available agent list.

Available agents:
{json.dumps(available_agents, indent=2)}

Rules:
- Return ONLY valid JSON.
- Do NOT explain anything.
- "chosen_agent" MUST exactly match one of the available agents.
- "memory_needed" should be true if previous conversation context is needed to answer properly.
- "better_prompt" should be rewritten clearly, with enough detail for the selected agent.
- Prefer specialized agents over generic ones.
- If the query relates to uploaded documents, PDFs, spreadsheets, retrieval, semantic search, or asks about previous uploaded files, choose the document-related agent.
- If the query involves calculations, analytics, financial insights, tables, statistics, or spreadsheet analysis, choose the appropriate analysis/tabulation agent.

Return format:
{{
    "memory_needed": true,
    "better_prompt": "rewritten prompt here",
    "chosen_agent": "AgentName"
}}
"""
        # log_print(f"RouterAgent system prompt: {system_prompt}")

        user_prompt = f"""
Current User Query:
{query}
"""
        intent_prompt = system_prompt + "\n\n" + user_prompt

        response = generate_chat_response(
            system_prompt,
            user_prompt,
            chat_history=memory_history,
            intent_search=True,
        )
        log_print(f"RouterAgent LLM response: {response}")
        intent_prompt = intent_prompt + " " + response
        try:
            routing_data = json.loads(response)

            chosen_agent_name = routing_data.get("chosen_agent")
            better_prompt = routing_data.get("better_prompt", query)
            memory_needed = routing_data.get("memory_needed", False)

            selected_agent = next(
                (
                    agent
                    for agent in self.agents
                    if agent.name == chosen_agent_name
                ),
                None,
            )

            if selected_agent is None:
                raise ValueError(
                    f"Invalid agent selected: {chosen_agent_name}"
                )

            log_print(
                f"RouterAgent selected: {chosen_agent_name}"
            )

            log_print(
                f"RouterAgent better prompt: {better_prompt}"
            )

            return selected_agent.handle_with_logging(
                memory_needed=memory_needed,
                query=better_prompt,
                intent_query=intent_prompt,
                request_id=request_id,
                session_id=session_id,
            )

        except Exception as e:
            log_print(f"RouterAgent LLM routing failed: {str(e)}")

            # fallback to semantic routing
            query_embedding = np.array(
                generate_embedding(query),
                dtype=np.float32
            )

            scored_agents = [
                (
                    agent,
                    self._agent_score(agent, query_embedding)
                )
                for agent in self.agents
            ]

            best_agent, _ = max(
                scored_agents,
                key=lambda item: item[1]
            )

            return best_agent.handle_with_logging(
                memory_needed=False,
                query=query,
                intent_query=intent_prompt,
                request_id=request_id,
                session_id=session_id,
            )