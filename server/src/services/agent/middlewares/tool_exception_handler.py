from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from server.src.services.agent.middlewares.base_middleware import BaseMiddleware

class ToolExceptionHandlerMiddleware(BaseMiddleware, AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        try:
            return handler(request)
        except Exception as e:
            return self.tool_message(e, request.tool_call["id"])

    async def awrap_tool_call(self, request, handler):
        try:
            return await handler(request)
        except Exception as e:
            return self.tool_message(e, request.tool_call["id"])

    def tool_message(self, e: Exception, tool_call_id: str):
        return ToolMessage(
            content=f"An error occurred while executing the tool: {e}, find another workaround and notify the user about this error.",
            tool_call_id=tool_call_id,
        )