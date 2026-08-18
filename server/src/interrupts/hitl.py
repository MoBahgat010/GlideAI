from langchain.agents.middleware import HumanInTheLoopMiddleware

SENSITIVE_TOOLS = {
    "send_email":               True,
    "gmail_send_email":         True,
    "gmail_create_draft":       True,
    "gmail_modify_message":     {"allowed_decisions": ["approve", "edit", "reject"]},
    "gmail_delete_message":     {"allowed_decisions": ["approve", "reject"]},
    "gmail_update_settings":    {"allowed_decisions": ["approve", "reject"]},
    
    "create_file":              True,
    "update_file":              True,
    "delete_file":              {"allowed_decisions": ["approve", "reject"]},
    "move_file":                {"allowed_decisions": ["approve", "edit", "reject"]},
    
    "update_spreadsheet":       True,
    "create_spreadsheet":       True,
    "delete_spreadsheet":       {"allowed_decisions": ["approve", "reject"]},
    
    "create_document":          True,
    "update_document":          True,
    
    "create_event":             True,
    "update_event":             True,
    "delete_event":             {"allowed_decisions": ["approve", "reject"]},
}


def build_hitl_middleware() -> HumanInTheLoopMiddleware:
    return HumanInTheLoopMiddleware(
        interrupt_on=SENSITIVE_TOOLS,
        description_prefix="Enterprise RAG: action requires your approval",
    )
