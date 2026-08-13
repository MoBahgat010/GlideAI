from langchain.agents.middleware import SummarizationMiddleware
from server.src.services.agent.middlewares.base_middleware import BaseMiddleware
from server.src.services.agent.models.summarizer import summarizer
from config import TOKEN_THRESHOLD

class CustomSummarizationMiddleware(BaseMiddleware, SummarizationMiddleware):
    def __init__(self, model=summarizer, trigger=("tokens", int(TOKEN_THRESHOLD))):
        super().__init__(model=model, trigger=trigger)