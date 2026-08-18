import asyncio
import datetime
import json
import logging
from typing import AsyncGenerator, Dict, Any, Optional, List

from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.types import Command
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage

from config import MONGODB_DB_NAME
from src.db.mongo import mongo
from src.interrupts import build_hitl_middleware
from .middlewares.base_middleware import BaseMiddleware
from .models.orchestrator import orchestrator
from .system_prompt import SYSTEM_PROMPT
from .tools.base_tools import Tools

logger = logging.getLogger("agent.workflow")


def _extract_citations(content: Any) -> List[dict]:
    """Parse citations from tool content (dict, JSON string, Pydantic model, or repr)."""
    if not content:
        return []

    raw_chunks = None
    if isinstance(content, dict):
        raw_chunks = content.get("chunks", [])
    elif isinstance(content, str):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "chunks" in parsed:
                raw_chunks = parsed["chunks"]
        except Exception:
            pass
    elif hasattr(content, "chunks"):
        raw_chunks = content.chunks

    if not raw_chunks or not isinstance(raw_chunks, list):
        return []

    citations = []
    for idx, c in enumerate(raw_chunks, 1):
        if hasattr(c, "model_dump"):
            c = c.model_dump()
        elif hasattr(c, "dict") and callable(c.dict):
            c = c.dict()
        elif not isinstance(c, dict):
            continue

        file_url = c.get("file_url") or c.get("url")
        citations.append({
            "index": c.get("index", idx),
            "custom_id": c.get("custom_id", f"doc_{idx}"),
            "file_name": c.get("file_name", "document.pdf"),
            "file_url": file_url,
            "url": file_url,
            "page": c.get("page", 1),
            "start_time": c.get("start_time"),
            "end_time": c.get("end_time"),
            "bbox": c.get("bbox"),
            "type": c.get("type", "text"),
            "score": c.get("score", 0.0),
            "text": c.get("text") or c.get("chunk_text") or "",
        })
    return citations


async def save_turn_to_mongodb(
    session_id: str,
    user_message: str,
    ai_reply: str,
    citations: List[dict],
) -> None:
    """Append user message and assistant reply with citations to MongoDB session document."""
    if not session_id or not ai_reply:
        return

    now = datetime.datetime.now(datetime.timezone.utc)
    new_messages = [
        {"role": "user", "content": user_message, "timestamp": now},
        {"role": "assistant", "content": ai_reply, "citations": citations, "timestamp": now},
    ]
    try:
        mongo.sessions.update_one(
            {"session_id": session_id},
            {
                "$push": {"messages": {"$each": new_messages}},
                "$set": {"updated_at": now},
            },
        )
        logger.info("Saved conversation turn to MongoDB for session %s (citations=%d)", session_id, len(citations))
    except Exception as exc:
        logger.warning("Failed to save turn to MongoDB for session %s: %s", session_id, exc)


class AgenticRAG:
    efficiency_system_prompt: str = SYSTEM_PROMPT

    def __init__(self, recursion_limit: int = 1000):
        self.orchestrator = orchestrator
        self.today_date = datetime.datetime.now().strftime("%Y-%m-%d")
        self.system_prompt = self.__class__.efficiency_system_prompt.format(today_date=self.today_date)
        self.middleware = BaseMiddleware.registry + [build_hitl_middleware()]
        self.recursion_limit = recursion_limit
        self.checkpointer = MongoDBSaver(mongo.sync_client, db_name=MONGODB_DB_NAME)
        self.tools = Tools.registry

        logger.info("Initializing AgenticRAG agent with %d tools: %s", len(self.tools), [getattr(t, "name", str(t)) for t in self.tools])
        self.agent: CompiledStateGraph = create_agent(
            model=self.orchestrator,
            system_prompt=self.system_prompt,
            tools=self.tools,
            middleware=self.middleware,
            checkpointer=self.checkpointer,
        )

    def _build_config(self, session_id: Optional[str], user_id: Optional[str]) -> Dict[str, Any]:
        """Construct runtime execution configuration for LangGraph."""
        return {
            "configurable": {
                "thread_id": session_id,
                "user_id": user_id,
            },
            "recursion_limit": self.recursion_limit,
        }

    async def astream_response(
        self,
        user_message: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream SSE events for a user prompt, strictly streaming final response tokens and emitting sources."""
        config = self._build_config(session_id, user_id)
        input_payload = {"messages": [HumanMessage(content=user_message)]}

        try:
            citations: List[dict] = []
            emitted_sources = False

            async for mode, chunk in self.agent.astream(
                input_payload,
                config=config,
                stream_mode=["messages", "updates"],
                context={"session_id": session_id, "user_id": user_id},
            ):
                if mode == "messages":
                    message, metadata = chunk
                    node_name = metadata.get("langgraph_node", "")

                    # Filter out ToolMessage or tools node output from token stream
                    if isinstance(message, ToolMessage) or getattr(message, "type", "") == "tool" or node_name == "tools":
                        found = _extract_citations(getattr(message, "content", None))
                        if found:
                            citations.extend(found)
                        continue

                    # Filter out intermediate tool calling chunks
                    if getattr(message, "tool_call_chunks", None) or getattr(message, "tool_calls", None):
                        continue

                    # Only stream text content from AI messages
                    is_ai = isinstance(message, (AIMessageChunk, AIMessage)) or getattr(message, "type", "") in ("ai", "AIMessageChunk")
                    if is_ai and message.content:
                        if citations and not emitted_sources:
                            yield {
                                "type": "sources",
                                "citations": citations,
                            }
                            emitted_sources = True

                        if isinstance(message.content, str):
                            yield {
                                "type": "token",
                                "content": message.content,
                            }
                        elif isinstance(message.content, list):
                            for block in message.content:
                                if isinstance(block, str) and block:
                                    yield {
                                        "type": "token",
                                        "content": block,
                                    }
                                elif isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                                    yield {
                                        "type": "token",
                                        "content": block["text"],
                                    }

                elif mode == "updates":
                    if "__interrupt__" in chunk:
                        interrupts = chunk["__interrupt__"]
                        action_requests = []
                        review_configs = []
                        if interrupts and len(interrupts) > 0:
                            int_obj = interrupts[0]
                            int_val = getattr(int_obj, "value", int_obj)
                            if isinstance(int_val, dict):
                                action_requests = int_val.get("action_requests", [])
                                review_configs = int_val.get("review_configs", [])
                                for req in action_requests:
                                    if "args" in req and "arguments" not in req:
                                        req["arguments"] = req["args"]
                                    if "arguments" in req and "args" not in req:
                                        req["args"] = req["arguments"]

                        serialized_interrupts = [getattr(i, "value", i) for i in interrupts] if interrupts else []
                        yield {
                            "type": "approval_required",
                            "data": serialized_interrupts,
                            "action_requests": action_requests,
                            "review_configs": review_configs,
                        }
                        return

                    # Extract citations from tools state in updates without leaking raw tools output
                    if "tools" in chunk:
                        t_data = chunk["tools"]
                        t_msgs = t_data.get("messages", []) if isinstance(t_data, dict) else (t_data if isinstance(t_data, list) else [])
                        for tm in t_msgs:
                            found = _extract_citations(getattr(tm, "content", None))
                            if found:
                                citations.extend(found)

            # After streaming finishes, retrieve result from state and save to MongoDB
            state = self.agent.get_state(config)
            if state and state.values:
                messages = state.values.get("messages", [])
                ai_reply = ""

                for msg in reversed(messages):
                    if hasattr(msg, "content") and msg.content and (getattr(msg, "type", "") == "ai" or msg.__class__.__name__.startswith("AIMessage")):
                        if isinstance(msg.content, str):
                            ai_reply = msg.content
                        elif isinstance(msg.content, list):
                            ai_reply = "".join(b if isinstance(b, str) else (b.get("text", "") if isinstance(b, dict) else "") for b in msg.content)
                        break

                if not citations:
                    for msg in messages:
                        if getattr(msg, "name", "") == "rag_retrieval" or getattr(msg, "tool_name", "") == "rag_retrieval" or getattr(msg, "type", "") == "tool":
                            found = _extract_citations(getattr(msg, "content", None))
                            if found:
                                citations.extend(found)

                if citations and not emitted_sources:
                    yield {
                        "type": "sources",
                        "citations": citations,
                    }
                    emitted_sources = True

                if session_id and ai_reply:
                    await save_turn_to_mongodb(session_id, user_message, ai_reply, citations)

        except Exception as exc:
            logger.exception("Error during stream execution: %s", exc)
            raise exc

    async def arun(
        self,
        user_message: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream plain text tokens for a given prompt."""
        async for event in self.astream_response(user_message, session_id, user_id=user_id):
            if event["type"] == "token":
                yield event["content"]

    async def resume(
        self,
        decisions: List,
        session_id: str,
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Resume agent execution after a human-in-the-loop interrupt decision."""
        config = self._build_config(session_id, user_id)
        payload = Command(resume={"decisions": decisions})
        async for mode, chunk in self.agent.astream(
            payload,
            config=config,
            stream_mode=["messages", "updates"],
            context={"session_id": session_id, "user_id": user_id},
        ):
            if mode == "messages":
                message, metadata = chunk
                node_name = metadata.get("langgraph_node", "")
                if isinstance(message, ToolMessage) or getattr(message, "type", "") == "tool" or node_name == "tools":
                    continue
                if getattr(message, "tool_call_chunks", None) or getattr(message, "tool_calls", None):
                    continue
                is_ai = isinstance(message, (AIMessageChunk, AIMessage)) or getattr(message, "type", "") in ("ai", "AIMessageChunk")
                if is_ai and message.content:
                    if isinstance(message.content, str):
                        yield message.content
                    elif isinstance(message.content, list):
                        for block in message.content:
                            if isinstance(block, str) and block:
                                yield block
                            elif isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                                yield block["text"]


agent_runner = AgenticRAG()