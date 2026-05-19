import time
import uuid

from services.logging_service import count_tokens, log_agent_call


class BaseAgent:
    name = "Base Agent"

    def can_handle(self, query: str) -> bool:
        raise NotImplementedError

    def score(self, query: str) -> float:
        return 0.0

    def handle(self, query: str, session_id: str | None = None):
        raise NotImplementedError

    def handle_with_logging(self, query: str, request_id: str | None = None, session_id: str | None = None):
        request_id = request_id or str(uuid.uuid4())
        start_time = time.time()

        try:
            response = self.handle(query, session_id=session_id)
        except Exception as exc:
            latency = time.time() - start_time
            log_agent_call(
                agent_name=self.name,
                request_id=request_id,
                query=query,
                response={"error": str(exc)},
                latency_seconds=latency,
                input_tokens=count_tokens(query),
                output_tokens=0,
                extra={"error_type": type(exc).__name__},
            )
            raise

        latency = time.time() - start_time
        answer_text = ""
        context = None
        if isinstance(response, dict):
            answer_text = str(response.get("answer", ""))
            if "_context" in response:
                context = response.pop("_context")
        elif isinstance(response, str):
            answer_text = response

        extra = {"context": context} if context is not None else None

        log_agent_call(
            agent_name=self.name,
            request_id=request_id,
            query=query,
            response=response,
            latency_seconds=latency,
            input_tokens=count_tokens(query),
            output_tokens=count_tokens(answer_text),
            extra=extra,
        )

        return response