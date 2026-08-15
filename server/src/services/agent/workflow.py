import os
import uuid
import logging
import datetime
from typing import AsyncGenerator, Dict, Any, Optional

from config import REDIS_URL

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langgraph.graph.state import CompiledStateGraph

from .models.orchestrator import orchestrator
from .system_prompt import SYSTEM_PROMPT
from .middlewares.base_middleware import BaseMiddleware
from .tools.base_tools import Tools
from .mcp.base_mcp import get_mcp_tools, McpSessions

logger = logging.getLogger("agent.workflow")


class AgenticRAG:
    def __init__(self, redis_url: str = REDIS_URL, recursion_limit: int = 1000):
        self.orchestrator = orchestrator
        self.today_date = datetime.datetime.now().strftime("%Y-%m-%d")
        self.system_prompt = SYSTEM_PROMPT.format(today_date=self.today_date)
        self.middleware = BaseMiddleware.registry
        self.recursion_limit = recursion_limit
        self.redis_url = redis_url

        self.mcp_session: McpSessions = None
        self.agent: CompiledStateGraph = None

    async def init_agent(self) -> CompiledStateGraph:
        if self.agent is not None:
            return self.agent

        mcp_tools, self.mcp_session = await get_mcp_tools()
        all_tools = Tools.registry + mcp_tools

        logger.info("Initializing Agentic RAG graph with %d tools and %d middlewares (Redis backend: %s)", len(all_tools), len(self.middleware), self.redis_url)
        self.agent = create_agent(
            model=self.orchestrator,
            system_prompt=self.system_prompt,
            tools=all_tools,
            middleware=self.middleware,
        )
        return self.agent

    def get_redis_history(self, session_id: str) -> RedisChatMessageHistory:
        return RedisChatMessageHistory(
            session_id=f"session:{session_id}:working_memory",
            url=self.redis_url,
            ttl=7 * 86400,
        )

    async def astream_response(self, user_message: str, session_id: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        await self.init_agent()

        active_thread_id = session_id or str(uuid.uuid4())
        config = {
            "configurable": {"thread_id": active_thread_id},
            "recursion_limit": self.recursion_limit,
        }

        # Load session history from LangChain Redis storage
        history_client = self.get_redis_history(active_thread_id)
        past_msgs = history_client.messages

        messages_input = list(past_msgs) + [{"role": "user", "content": user_message}]
        input_payload = {"messages": messages_input}

        logger.info("Executing Agentic RAG for session_id=%s with %d prior messages from Redis", active_thread_id, len(past_msgs))
        collected_tokens = []

        async for event in self.agent.astream_events(input_payload, config=config, version="v2"):
            kind = event.get("event")

            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    token_text = str(chunk.content)
                    collected_tokens.append(token_text)
                    yield {"type": "token", "content": token_text}

            elif kind == "on_tool_start":
                tool_name = event.get("name", "tool")
                tool_input = event.get("data", {}).get("input", "")
                yield {
                    "type": "step",
                    "name": tool_name,
                    "input": str(tool_input)[:300],
                }

            elif kind == "on_tool_end":
                tool_name = event.get("name", "tool")
                tool_output = event.get("data", {}).get("output", "")
                if hasattr(tool_output, "content"):
                    out_text = tool_output.content
                elif isinstance(tool_output, list):
                    out_text = "\n".join(str(x) for x in tool_output)
                else:
                    out_text = str(tool_output)

                yield {
                    "type": "step_result",
                    "name": tool_name,
                    "output": out_text[:500] if len(out_text) > 500 else out_text,
                }

                if tool_name == "rag_retrieval" and "--- Result" in out_text:
                    citations = []
                    chunks_raw = out_text.split("--- Result ")
                    for cr in chunks_raw:
                        if not cr.strip():
                            continue
                        c_dict = {}
                        first_line = cr.split("\n", 1)[0]
                        if "---" in first_line:
                            try:
                                c_dict["index"] = int(first_line.replace("---", "").strip())
                            except Exception:
                                pass

                        for line in cr.split("\n"):
                            line = line.strip()
                            if line.startswith("Chunk ID:"):
                                c_dict["custom_id"] = line.split(":", 1)[1].strip()
                            elif line.startswith("File Name:"):
                                c_dict["file_name"] = line.split(":", 1)[1].strip()
                            elif line.startswith("Page:"):
                                try:
                                    c_dict["page"] = int(line.split(":", 1)[1].strip())
                                except Exception:
                                    pass
                            elif line.startswith("Time:"):
                                c_dict["time_range"] = line.split(":", 1)[1].strip()
                            elif line.startswith("BBox:"):
                                c_dict["bbox"] = line.split(":", 1)[1].strip()
                            elif line.startswith("Type:"):
                                c_dict["type"] = line.split(":", 1)[1].strip()
                            elif line.startswith("Rerank Score:"):
                                try:
                                    c_dict["score"] = float(line.split(":", 1)[1].strip())
                                except Exception:
                                    pass
                            elif line.startswith("Content:"):
                                c_dict["text"] = cr.split("Content:\n", 1)[-1].strip()

                        if c_dict.get("file_name"):
                            if "index" not in c_dict:
                                c_dict["index"] = len(citations) + 1
                            citations.append(c_dict)
                    if citations:
                        yield {"type": "sources", "citations": citations}

        # Persist conversation turn to LangChain Redis history
        try:
            history_client.add_user_message(user_message)
            if collected_tokens:
                history_client.add_ai_message("".join(collected_tokens))
        except Exception as exc:
            logger.warning("Failed to append conversation turn to Redis history: %s", exc)

    async def arun(self, user_message: str, session_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        async for event in self.astream_response(user_message, session_id):
            if event["type"] == "token":
                yield event["content"]

    def cleanup(self):
        if self.mcp_session:
            self.mcp_session.stop()