import json
import os
import re
import logging
from asyncio import Event, create_task, wait, FIRST_COMPLETED
from contextlib import AsyncExitStack
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv()

logger = logging.getLogger("agent.mcp")

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}|\{([A-Z0-9_]+)\}")


def _resolve_env_vars(obj):
    """Recursively resolve ${VAR} or {VAR} placeholders in JSON values using os.environ."""
    if isinstance(obj, str):
        def replacer(m):
            var_name = m.group(1) or m.group(2)
            return os.environ.get(var_name, m.group(0))
        return _ENV_VAR_PATTERN.sub(replacer, obj)
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    return obj


def mcp_connections() -> dict:
    """Load MCP server definitions from mcp_connections.json with environment variable substitution."""
    config_path = os.path.join(os.path.dirname(__file__), "mcp_connections.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data:
                return _resolve_env_vars(data)
        except Exception as e:
            logger.warning("Failed to load mcp_connections.json: %s", e)

    smithery_api_key = os.getenv("SMITHERY_API_KEY", "")
    if smithery_api_key:
        return {
            "gmail": {
                "transport": "stdio",
                "command": "npx",
                "args": [
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "gmail",
                    "--config",
                    f'{{"smitheryApiKey":"{smithery_api_key}"}}',
                ],
            }
        }
    return {}


class McpSessions:
    """Manages persistent MCP server sessions using AsyncExitStack with fault-tolerance and clean lifecycle."""

    def __init__(self, connections: dict = None):
        self.connections = connections if connections is not None else mcp_connections()
        self.tools = []
        self._ready = Event()
        self._stop = Event()
        self._task = None

    async def _run(self):
        if not self.connections:
            self._ready.set()
            return

        try:
            client = MultiServerMCPClient(self.connections)
            async with AsyncExitStack() as stack:
                for name in list(self.connections.keys()):
                    try:
                        session = await stack.enter_async_context(client.session(name))
                        loaded = await load_mcp_tools(session, server_name=name)
                        self.tools.extend(loaded)
                        logger.info("Successfully loaded %d tools from MCP server '%s'", len(loaded), name)
                    except Exception as exc:
                        logger.error("Failed to initialize MCP server '%s': %s", name, exc)
                self._ready.set()
                if self.tools:
                    await self._stop.wait()
        except Exception as exc:
            logger.error("Error in MCP session runner: %s", exc)
            self._ready.set()

    async def start(self) -> list:
        if not self.connections:
            return []
        try:
            self._task = create_task(self._run())
            ready = create_task(self._ready.wait())
            await wait([ready, self._task], return_when=FIRST_COMPLETED)
            ready.cancel()
            if self._task.done() and not self._task.cancelled():
                exc = self._task.exception()
                if exc:
                    logger.error("MCP session task encountered exception: %s", exc)
            if not self.tools:
                self.stop()
            return self.tools
        except Exception as exc:
            logger.error("Error starting MCP sessions: %s", exc)
            self.stop()
            return []

    def stop(self):
        self._stop.set()


async def get_mcp_tools():
    """Instantiate MCP sessions and return (tools, session_handle)."""
    connections = mcp_connections()
    if not connections:
        return [], None
    try:
        sessions = McpSessions(connections)
        mcp_tools = await sessions.start()
        if not mcp_tools:
            sessions.stop()
            return [], None
        return mcp_tools, sessions
    except Exception as exc:
        logger.error("Failed in get_mcp_tools: %s", exc)
        return [], None