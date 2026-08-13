from __future__ import annotations
from typing import List
from langchain_core.tools import BaseTool


class Tools:
    cls_tool: BaseTool
    registry: List[BaseTool] = []

    def __init_subclass__(cls):
        try:
            Tools.registry.append(cls().cls_tool)
        except Exception:
            pass