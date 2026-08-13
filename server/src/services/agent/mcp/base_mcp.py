import json
import os
import logging
from asyncio import Event, create_task, wait, FIRST_COMPLETED
from contextlib import AsyncExitStack
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

logger = logging.getLogger("agent.mcp")

def mcp_connections() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "mcp_connections.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load mcp_connections.json: %s", e)
    return {}

class McpSessions:
    def __init__(self, connections: dict = None):
        self.connections = connections if connections is not None else mcp_connections()
        self.tools = []
        self._ready = Event()
        self._stop = Event()
        self._task = None

    async def _run(self):
        if not self.connections:
            self._ready.set()
            await self._stop.wait()
            return

        client = MultiServerMCPClient(self.connections)
        async with AsyncExitStack() as stack:
            for name in self.connections:
                session = await stack.enter_async_context(client.session(name))
                self.tools += await load_mcp_tools(session, server_name=name)
            self._ready.set()
            await self._stop.wait()

    async def start(self) -> list:
        if not self.connections:
            return []
        self._task = create_task(self._run())
        ready = create_task(self._ready.wait())
        await wait([ready, self._task], return_when=FIRST_COMPLETED)
        ready.cancel()
        if self._task.done():
            self._task.result()
        return self.tools

    def stop(self):
        self._stop.set()

async def get_mcp_tools():
    connections = mcp_connections()
    if not connections:
        return [], None
    sessions = McpSessions(connections)
    mcp_tools = await sessions.start()
    return mcp_tools, sessions