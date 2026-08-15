from langchain.agents.middleware import ModelCallLimitMiddleware
from .base_middleware import BaseMiddleware
from config import RUN_LIMIT

class CustomModelCallLimitMiddleware(BaseMiddleware, ModelCallLimitMiddleware):
    def __init__(self, exit_behavior='end', run_limit=int(RUN_LIMIT)):
        super().__init__(exit_behavior=exit_behavior, run_limit=run_limit)