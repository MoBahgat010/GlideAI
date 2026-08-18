import asyncio
import datetime
import json
import logging
from typing import AsyncGenerator, Dict, Any, Optional, List

from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.types import Command
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage

from config import MONGODB_URL, MONGODB_DB_NAME
from src.db.mongo import mongo_client
from src.interrupts import build_hitl_middleware
from src.models.schemas import ChunkMetadata
from .middlewares.base_middleware import BaseMiddleware
from .models.orchestrator import orchestrator
from .system_prompt import SYSTEM_PROMPT
from .tools.base_tools import Tools

logger = logging.getLogger("agent.workflow")


def parse_rag_citations(tool_output: Any, existing_citations: List[dict]) -> List[dict]:
    """Parse structured citation metadata from RAG retrieval tool output."""
    if not isinstance(tool_output, str) or "--- Result" not in tool_output:
        return existing_citations

    chunks_raw = tool_output.split("--- Result ")
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
            elif line.startswith("File URL:"):
                f_url = line.split(":", 1)[1].strip()
                if f_url and f_url != "None":
                    c_dict["file_url"] = f_url
                    c_dict["url"] = f_url
            elif line.startswith("Page:"):
                try:
                    c_dict["page"] = int(line.split(":", 1)[1].strip())
                except Exception:
                    pass
            elif line.startswith("Time:"):
                time_str = line.split(":", 1)[1].strip()
                c_dict["time_range"] = time_str
                try:
                    parts = time_str.replace("s", "").split("-")
                    if len(parts) == 2:
                        c_dict["start_time"] = float(parts[0].strip())
                        c_dict["end_time"] = float(parts[1].strip())
                except Exception:
                    pass
            elif line.startswith("BBox:"):
                raw_bbox = line.split(":", 1)[1].strip()
                try:
                    c_dict["bbox"] = json.loads(raw_bbox)
                except Exception:
                    try:
                        c_dict["bbox"] = [
                            float(x.strip())
                            for x in raw_bbox.strip("[]").split(",")
                            if x.strip()
                        ]
                    except Exception:
                        c_dict["bbox"] = None
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
                c_dict["index"] = len(existing_citations) + 1
            if not c_dict.get("url"):
                c_dict["url"] = c_dict.get("file_url")

            try:
                pydantic_cite = ChunkMetadata(**c_dict)
                existing_citations.append(pydantic_cite.model_dump())
            except Exception:
                existing_citations.append(c_dict)

    return existing_citations


async def save_turn_to_mongodb(
    db: Any,
    session_id: str,
    user_message: str,
    ai_reply: str,
    citations: List[dict],
) -> None:
    """Append user message and assistant reply with citations to MongoDB session document."""
    if db is None or not session_id or not ai_reply:
        return

    now = datetime.datetime.now(datetime.timezone.utc)
    new_messages = [
        {"role": "user", "content": user_message, "timestamp": now},
        {"role": "assistant", "content": ai_reply, "citations": citations, "timestamp": now},
    ]
    try:
        await db.sessions.update_one(
            {"session_id": session_id},
            {
                "$push": {"messages": {"$each": new_messages}},
                "$set": {"updated_at": now},
            },
        )
        logger.info("Saved conversation turn to MongoDB for session %s", session_id)
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
        self.checkpointer = MongoDBSaver(mongo_client, db_name=MONGODB_DB_NAME)
        self.tools = Tools.registry

        logger.info("Tool Names: %s", [getattr(t, "name", str(t)) for t in self.tools])

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
        db: Optional[Any] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream SSE events for a user prompt, handling tokens, tool steps, citations, and HiTL interrupts."""
        config = self._build_config(session_id, user_id)
        input_payload = {"messages": [HumanMessage(content=user_message)]}
        collected_tokens: List[str] = []
        citations: List[dict] = []

        try:
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
                        out_text = str(tool_output.content)
                    elif isinstance(tool_output, list):
                        out_text = "\n".join(str(x) for x in tool_output)
                    else:
                        out_text = str(tool_output)

                    yield {
                        "type": "step_result",
                        "name": tool_name,
                        "output": out_text[:500] if len(out_text) > 500 else out_text,
                    }

                    if tool_name == "rag_retrieval":
                        parse_rag_citations(out_text, citations)
                        if citations:
                            yield {"type": "sources", "citations": citations}

                elif kind == "on_chain_stream":
                    chunk = event.get("data", {}).get("chunk", {})
                    if isinstance(chunk, dict) and "__interrupt__" in chunk:
                        int_tuple = chunk["__interrupt__"]
                        if int_tuple and len(int_tuple) > 0:
                            int_obj = int_tuple[0]
                            int_val = getattr(int_obj, "value", int_obj)
                            if isinstance(int_val, dict):
                                action_requests = int_val.get("action_requests", [])
                                for req in action_requests:
                                    if "args" in req and "arguments" not in req:
                                        req["arguments"] = req["args"]
                                    if "arguments" in req and "args" not in req:
                                        req["args"] = req["arguments"]
                                logger.info("Yielding approval_required from on_chain_stream for session %s", session_id)
                                yield {
                                    "type": "approval_required",
                                    "action_requests": action_requests,
                                    "review_configs": int_val.get("review_configs", []),
                                }
                                return

                elif kind == "on_interrupt":
                    interrupt_value = event.get("data", {}).get("value", {})
                    action_requests = interrupt_value.get("action_requests", [])
                    for req in action_requests:
                        if "args" in req and "arguments" not in req:
                            req["arguments"] = req["args"]
                        if "arguments" in req and "args" not in req:
                            req["args"] = req["arguments"]
                    logger.info("Yielding approval_required from on_interrupt for session %s", session_id)
                    yield {
                        "type": "approval_required",
                        "action_requests": action_requests,
                        "review_configs": interrupt_value.get("review_configs", []),
                    }
                    return

            # Check if execution paused at an interrupt
            state = self.agent.get_state(config)
            if state and state.tasks:
                for task in state.tasks:
                    if hasattr(task, "interrupts") and task.interrupts:
                        for intr in task.interrupts:
                            int_val = getattr(intr, "value", intr)
                            if isinstance(int_val, dict):
                                action_requests = int_val.get("action_requests", [])
                                for req in action_requests:
                                    if "args" in req and "arguments" not in req:
                                        req["arguments"] = req["args"]
                                    if "arguments" in req and "args" not in req:
                                        req["args"] = req["arguments"]
                                logger.info("Yielding approval_required from agent state for session %s", session_id)
                                yield {
                                    "type": "approval_required",
                                    "action_requests": action_requests,
                                    "review_configs": int_val.get("review_configs", []),
                                }
                                return

            ai_reply = "".join(collected_tokens)
            if db is not None and session_id:
                await save_turn_to_mongodb(db, session_id, user_message, ai_reply, citations)

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
        async for event in self.agent.astream_events(payload, config=config, version="v2"):
            if event.get("event") == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield str(chunk.content)


agent_runner = AgenticRAG()