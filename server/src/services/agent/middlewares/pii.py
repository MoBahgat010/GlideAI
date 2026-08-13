from typing import Any
from langchain.agents import AgentState
from langchain.agents.middleware import PIIMiddleware
from langchain_core.messages import SystemMessage

class CustomPIIMiddleware(PIIMiddleware):
    system_message_content = (
        "WARNING: Sensitive PII of type '{pii_type}' was detected and redacted from the input. "
        "Please explicitly warn the user that PII of type '{pii_type}' was detected and redacted."
    )

    def __init__(self, pii_type: str, strategy: str = "redact", apply_to_tool_results: bool = True):
        super().__init__(pii_type=pii_type, strategy=strategy, apply_to_tool_results=apply_to_tool_results)

    def before_model(self, state: AgentState, runtime):
        result = super().before_model(state, runtime)
        if result is not None and isinstance(result, dict) and "messages" in result:
            new_messages = list(result["messages"])
            new_messages.append(SystemMessage(content=self.system_message_content.format(pii_type=self.pii_type)))
            return {"messages": new_messages}
        return result

    async def abefore_model(self, state: AgentState, runtime):
        result = await super().abefore_model(state, runtime)
        if result is not None and isinstance(result, dict) and "messages" in result:
            new_messages = list(result["messages"])
            new_messages.append(SystemMessage(content=self.system_message_content.format(pii_type=self.pii_type)))
            return {"messages": new_messages}
        return result
