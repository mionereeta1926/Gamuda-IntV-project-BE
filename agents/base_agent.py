class BaseAgent:
    name = "Base Agent"

    def can_handle(self, query: str) -> bool:
        raise NotImplementedError

    def handle(self, query: str):
        raise NotImplementedError