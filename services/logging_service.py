import json
import logging
from datetime import datetime
from pathlib import Path

try:
    import tiktoken
except ImportError:
    tiktoken = None

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_LOG_PATH = LOG_DIR / "output_log.jsonl"
PRINT_LOG_PATH = LOG_DIR / "print_log.log"

output_logger = logging.getLogger("project_intelligence.output")
output_logger.setLevel(logging.INFO)
output_logger.propagate = False

if not output_logger.handlers:
    output_handler = logging.FileHandler(OUTPUT_LOG_PATH, encoding="utf-8")
    output_handler.setFormatter(logging.Formatter("%(message)s"))
    output_logger.addHandler(output_handler)

print_logger = logging.getLogger("project_intelligence.print")
print_logger.setLevel(logging.INFO)
print_logger.propagate = False

if not print_logger.handlers:
    print_handler = logging.FileHandler(PRINT_LOG_PATH, encoding="utf-8")
    print_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    print_logger.addHandler(print_handler)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    if not text:
        return 0

    if tiktoken is None:
        return len(text.split())

    try:
        encoding = tiktoken.encoding_for_model(encoding_name)
    except Exception:
        try:
            encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            return len(text.split())

    return len(encoding.encode(text))


def log_agent_call(agent_name: str, request_id: str, query: str, response: dict, latency_seconds: float, input_tokens: int, output_tokens: int, extra: dict | None = None):
    context = None
    if extra and "context" in extra:
        context = extra.pop("context")
    elif isinstance(response, dict) and "context" in response:
        context = response.pop("context")

    INPUT_COST_PER_MILLION = 0.15
    OUTPUT_COST_PER_MILLION = 0.60

    input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION

    token_cost = round(input_cost + output_cost, 8)

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "request_id": request_id,
        "agent": agent_name,
        "query": query,
        "response": response,
        "latency_ms": int(latency_seconds * 1000),

        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,

        "input_token_cost_usd": round(input_cost, 8),
        "output_token_cost_usd": round(output_cost, 8),
        "token_cost_usd": token_cost,

        "model": "openai/gpt-oss-120b",
    }

    if context is not None:
        payload["context"] = context

    if extra:
        payload.update(extra)

    output_logger.info(json.dumps(payload, ensure_ascii=False))


def log_print(*args, **kwargs):
    message = " ".join(str(arg) for arg in args)
    print_logger.info(message)
