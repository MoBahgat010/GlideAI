import os
import uuid
import logging
import datetime
from typing import AsyncGenerator, List, Union

from config import REDIS_URL
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from deepagents.backends import LocalShellBackend
from server.src.services.agent.models.orchestrator import orchestrator
from server.src.services.agent.system_prompt import SYSTEM_PROMPT
from server.src.services.agent.middlewares.base_middleware import BaseMiddleware
from server.src.services.agent.tools.base_tools import Tools
# Explicitly import tools to ensure registration
import server.src.services.agent.tools.RAG_retrieval
import server.src.services.agent.tools.document_summarizer
import server.src.services.agent.tools.python_calculator

from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from server.src.services.agent.mcp.base_mcp import get_mcp_tools, McpSessions

load_dotenv(override=True)
logger = logging.getLogger("agent.workflow")


class AgenticRAG:
    def __init__(self, redis_url: str = REDIS_URL, recursion_limit: int = 1000):
        self.orchestrator = orchestrator
        self.today_date = datetime.datetime.now().strftime("%Y-%m-%d")
        self.system_prompt = SYSTEM_PROMPT.format(today_date=self.today_date)
        self.middleware = BaseMiddleware.registry
        self.recursion_limit = recursion_limit

        try:
            logger.info("Connecting Redis checkpointer for Agentic RAG working memory: %s", redis_url)
            self.checkpointer = RedisSaver.from_conn_info(url=redis_url)
        except Exception as exc:
            logger.warning("Redis checkpointer failed to initialize (%s), falling back to InMemorySaver", exc)
            self.checkpointer = InMemorySaver()

        self.mcp_session: McpSessions = None
        self.agent: CompiledStateGraph = None

    async def init_agent(self) -> CompiledStateGraph:
        if self.agent is not None:
            return self.agent

        mcp_tools, self.mcp_session = await get_mcp_tools()
        all_tools = Tools.registry + mcp_tools

        logger.info("Initializing Agentic RAG graph with %d tools and %d middlewares", len(all_tools), len(self.middleware))
        self.agent = create_agent(
            model=self.orchestrator,
            system_prompt=self.system_prompt,
            tools=all_tools,
            middleware=self.middleware,
            checkpointer=self.checkpointer,
        )
        return self.agent

    async def arun(self, user_message: str, session_id: str = None) -> AsyncGenerator[str, None]:
        """
        Stream agent events token by token for stateless API servers.
        Loads state from Redis using session_id as the thread_id.
        """
        await self.init_agent()

        active_thread_id = session_id or str(uuid.uuid4())
        config = {
            "configurable": {"thread_id": active_thread_id},
            "recursion_limit": self.recursion_limit,
        }

        logger.info("Running Agentic RAG for session_id=%s", active_thread_id)
        input_payload = {"messages": [{"role": "user", "content": user_message}]}

        async for event in self.agent.astream_events(input_payload, config=config, version="v2"):
            kind = event.get("event")
            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield chunk.content
            elif kind == "on_tool_start":
                tool_name = event.get("name")
                tool_input = event.get("data", {}).get("input", "")
                yield f"\n\n**Executing Tool:** `{tool_name}` with input: `{tool_input}`...\n\n"
            elif kind == "on_tool_end":
                tool_name = event.get("name")
                tool_output = event.get("data", {}).get("output", "")
                if hasattr(tool_output, "content"):
                    output_str = tool_output.content
                elif isinstance(tool_output, list):
                    output_str = "\n".join(
                        item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in tool_output
                    )
                else:
                    output_str = str(tool_output)
                yield f"**Tool `{tool_name}` finished:**\n```\n{output_str}\n```\n\n"

    async def resume(self, decisions: List, session_id: str) -> AsyncGenerator[str, None]:
        await self.init_agent()
        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": self.recursion_limit,
        }
        payload = Command(resume={"decisions": decisions})
        async for token in self.agent.astream_events(payload, config=config, version="v2"):
            kind = token.get("event")
            if kind == "on_chat_model_stream":
                chunk = token.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield chunk.content

    def cleanup(self):
        if self.mcp_session:
            self.mcp_session.stop()