import os
from asyncio import Event, create_task, wait, FIRST_COMPLETED
from contextlib import AsyncExitStack
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

def mcp_connections() -> dict:
    smithery_api_key = os.getenv("SMITHERY_API_KEY", "")
    return {
        "gmail": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@smithery/cli@latest", "run", "gmail", "--config", f"{{\"smitheryApiKey\":\"{smithery_api_key}\"}}"],
        }
    }


class McpSessions():
    def __init__(self, connections: dict = mcp_connections()):
        self.connections = connections
        self.tools = []
        self._ready = Event()
        self._stop = Event()
        self._task = None

    async def _run(self):
        client = MultiServerMCPClient(self.connections)
        async with AsyncExitStack() as stack:
            for name in self.connections:
                session = await stack.enter_async_context(client.session(name))
                self.tools += await load_mcp_tools(session, server_name=name)
            self._ready.set()
            await self._stop.wait()

    async def start(self) -> list:
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
    sessions = McpSessions(mcp_connections())
    mcp_tools = await sessions.start()
    return mcp_tools, sessions
