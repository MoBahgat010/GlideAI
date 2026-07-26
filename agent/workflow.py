from langchain_openai import ChatOpenAI
import os
import uuid
import datetime
from typing import List
from dotenv import load_dotenv

from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from deepagents.backends import LocalShellBackend

from agent.mcp.base_mcp import get_mcp_tools, McpSessions
from agent.tools.rag_tool import query_knowledge_base

load_dotenv(override=True)

ORCHESTRATOR_MODEL=os.getenv("ORCHESTRATOR")
BASE_URL=os.getenv("BASE_URL")
API_KEY=os.getenv("NVIDIA_API_KEY")

class CoodingAgent:
    efficiency_system_prompt = """Today Date: {today_date}
    You are an intelligent agent that acts as an interface to the user's RAG knowledge base and their Gmail account.
    
    ## Tools
    - **query_knowledge_base**: Use this tool to search the uploaded documents for answers to user questions.
    - **Gmail MCP tools**: Use these tools to connect to and interact with the user's Gmail.
    
    When asked a question about documents, always use `query_knowledge_base`.
    When asked about emails, always use the Gmail tools.
    """

    def __init__(self, recursion_limit: int = 1000):
        # Using Gemini as the orchestrator since it supports robust tool calling
        self.orchestrator = ChatOpenAI(
            model=ORCHESTRATOR_MODEL,
            base_url=BASE_URL,
            api_key=API_KEY,
            top_p=0.92,
        )
        self.system_prompt = self.__class__.efficiency_system_prompt.format(today_date=datetime.datetime.now().strftime("%Y-%m-%d"))
        
        # We don't necessarily need the middlewares from reference unless explicitly requested.
        self.middleware = []
        self.checkpointer = InMemorySaver()
        self.backend = LocalShellBackend(root_dir=os.getcwd(), virtual_mode=True, inherit_env=True)

        self.config = {"configurable": {"thread_id": uuid.uuid4()}, "recursion_limit": recursion_limit}
        self.pending_actions = 0

        self.mcp_session: McpSessions = None
        self.coding_agent: CompiledStateGraph = None

    async def init_agent(self) -> CompiledStateGraph:
        mcp_tools, self.mcp_session = await get_mcp_tools()
        self.coding_agent = create_deep_agent(
            model=self.orchestrator,
            system_prompt=self.system_prompt,
            tools=[query_knowledge_base] + mcp_tools,
            middleware=self.middleware,
            backend=self.backend,
            checkpointer=self.checkpointer,
        )
    
    async def arun(self, messages: List):
        async for event in self.coding_agent.astream_events({ "messages": messages }, config=self.config, version="v2"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content"):
                    if isinstance(chunk.content, str):
                        yield chunk.content
                    elif isinstance(chunk.content, list):
                        for c in chunk.content:
                            if isinstance(c, dict) and c.get("type") == "text":
                                yield c.get("text", "")
                            elif isinstance(c, str):
                                yield c
            elif kind == "on_tool_start":
                tool_name = event.get("name")
                tool_input = event.get("data", {}).get("input", "")
                yield f"\n\n **Executing Tool:** `{tool_name}` with args: `{tool_input}`...\n\n"
            elif kind == "on_tool_end":
                tool_name = event.get("name")
                tool_output = event.get("data", {}).get("output", "")
                if hasattr(tool_output, "content"):
                    output_str = tool_output.content
                elif isinstance(tool_output, list):
                    output_str = "\n".join(item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in tool_output)
                else:
                    output_str = str(tool_output)
                # Keep output brief in chat history to avoid blowing up the UI
                if len(output_str) > 500:
                    output_str = output_str
                yield f" **Tool `{tool_name}` finished:**\n```\n{output_str}\n```\n\n"

    def cleanup(self):
        if self.mcp_session:
            self.mcp_session.stop()
