from typing import List
from langchain.agents.middleware import AgentMiddleware
from .pii import CustomPIIMiddleware

class BaseMiddleware(AgentMiddleware):
    registry: List[AgentMiddleware] = [
        CustomPIIMiddleware("email", strategy="redact"),
        CustomPIIMiddleware("credit_card", strategy="redact"),
        CustomPIIMiddleware("ip", strategy="redact"),
        CustomPIIMiddleware("mac_address", strategy="redact"),
    ]

    def __init_subclass__(cls):
        super().__init_subclass__()
        try:
            BaseMiddleware.registry.append(cls())
        except TypeError:
            pass