import asyncio
import base64
import logging
from email.message import EmailMessage
from typing import Optional, List, Dict, Any, Tuple

from googleapiclient.discovery import build
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from src.db.mongo import mongo_db
from src.services.google_creds import get_user_google_credentials
from .base_tools import Tools

logger = logging.getLogger("agent.tools.gmail")


from langchain_core.runnables.config import var_child_runnable_config


def resolve_user_id(config: Optional[RunnableConfig] = None) -> Optional[str]:
    """
    Extract user_id from:
    1. Direct RunnableConfig parameter
    2. LangChain active runtime context via var_child_runnable_config.get()
    3. MongoDB session lookup by thread_id
    """
    cfg_to_check = config
    if not cfg_to_check:
        try:
            cfg_to_check = var_child_runnable_config.get()
        except Exception:
            cfg_to_check = None

    if cfg_to_check and isinstance(cfg_to_check, dict):
        cfg = cfg_to_check.get("configurable", {})
        if isinstance(cfg, dict):
            # 1. Direct user_id
            if cfg.get("user_id"):
                return str(cfg["user_id"])

            # 2. Direct user
            if cfg.get("user"):
                return str(cfg["user"])

            # 3. Lookup via thread_id / session_id in MongoDB
            thread_id = cfg.get("thread_id") or cfg.get("session_id")
            if thread_id:
                try:
                    session_doc = mongo_db.sessions.find_one({"session_id": str(thread_id)})
                    if session_doc and session_doc.get("user_id"):
                        return str(session_doc["user_id"])
                except Exception as exc:
                    logger.debug("Failed to resolve user_id from thread_id %s: %s", thread_id, exc)

    return None


async def get_authenticated_gmail_service(config: Optional[RunnableConfig] = None) -> Tuple[Optional[Any], Optional[str]]:
    """
    Resolve user context and construct an authenticated Google Gmail API service.
    Returns (service, error_message).
    """
    user_id = resolve_user_id(config)
    if not user_id:
        return None, "Error: User ID not available in session context. Please ensure you are logged in."

    creds = await get_user_google_credentials(user_id)
    if not creds:
        return None, "The user has not connected their Google/Gmail account yet. Please connect Gmail in account settings."

    try:
        service = await asyncio.to_thread(
            lambda: build("gmail", "v1", credentials=creds, cache_discovery=False)
        )
        return service, None
    except Exception as exc:
        logger.exception("Failed to build Gmail service for user %s: %s", user_id, exc)
        return None, f"Failed to authenticate with Google Gmail service: {str(exc)}"


def parse_message_payload(msg_data: dict) -> dict:
    """Extract standard email headers and plain text content from a Gmail API message."""
    headers_list = msg_data.get("payload", {}).get("headers", [])
    headers = {h.get("name", "").lower(): h.get("value", "") for h in headers_list}

    body = ""
    payload = msg_data.get("payload", {})
    if "parts" in payload:
        for part in payload["parts"]:
            mime_type = part.get("mimeType", "")
            data = part.get("body", {}).get("data")
            if mime_type == "text/plain" and data:
                try:
                    body = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
                    break
                except Exception:
                    pass
            elif mime_type == "text/html" and data and not body:
                try:
                    body = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
                except Exception:
                    pass
    elif "body" in payload and "data" in payload["body"]:
        try:
            body = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")
        except Exception:
            pass

    if not body:
        body = msg_data.get("snippet", "")

    return {
        "id": msg_data.get("id"),
        "threadId": msg_data.get("threadId"),
        "subject": headers.get("subject", "No Subject"),
        "from": headers.get("from", "Unknown Sender"),
        "to": headers.get("to", ""),
        "date": headers.get("date", ""),
        "snippet": msg_data.get("snippet", ""),
        "body": body[:2000].strip() if len(body) > 2000 else body.strip(),
    }


@tool
async def fetch_user_emails(
    query: str = "is:inbox",
    max_results: int = 10,
    config: Optional[RunnableConfig] = None,
) -> str:
    """
    Fetch and search user emails from Gmail using the Google Gmail API.

    Args:
        query: Gmail search query filter (e.g. 'is:unread', 'from:colleague@company.com', 'subject:invoice', 'newer_than:7d', 'is:inbox').
        max_results: Maximum number of emails to retrieve (default: 10, max: 25).
        config: Execution runtime config containing user authentication info.

    Returns:
        Structured summary of emails including ID, Subject, From, Date, and message preview.
    """
    service, err = await get_authenticated_gmail_service(config)
    if err:
        return err

    try:
        def _fetch_sync():
            results = service.users().messages().list(
                userId="me",
                q=query,
                maxResults=min(max_results, 25),
            ).execute()
            messages = results.get("messages", [])

            email_summaries = []
            for msg_item in messages:
                msg_id = msg_item["id"]
                msg_data = service.users().messages().get(
                    userId="me",
                    id=msg_id,
                    format="full",
                ).execute()
                email_summaries.append(parse_message_payload(msg_data))
            return email_summaries

        email_list = await asyncio.to_thread(_fetch_sync)

        if not email_list:
            return f"No emails found matching the query: '{query}'"

        formatted_output = [f"Found {len(email_list)} email(s) for query: '{query}'\n"]
        for idx, email_item in enumerate(email_list, 1):
            snippet_text = email_item['snippet'] or email_item['body'][:150]
            entry = (
                f"### Email {idx}: {email_item['subject']}\n"
                f"- **Message ID**: `{email_item['id']}`\n"
                f"- **From**: {email_item['from']}\n"
                f"- **To**: {email_item['to']}\n"
                f"- **Date**: {email_item['date']}\n"
                f"- **Summary**: {snippet_text}\n"
            )
            formatted_output.append(entry)

        return "\n".join(formatted_output)

    except Exception as exc:
        logger.exception("Error executing fetch_user_emails: %s", exc)
        return f"Failed to fetch emails via Google API: {str(exc)}"


@tool
async def get_email_details(
    message_id: str,
    config: Optional[RunnableConfig] = None,
) -> str:
    """
    Get the full details and body content of a specific Gmail email by message ID.

    Args:
        message_id: The unique Gmail message ID.
        config: Execution runtime config containing user authentication info.

    Returns:
        Full details and readable text body of the email.
    """
    service, err = await get_authenticated_gmail_service(config)
    if err:
        return err

    try:
        def _get_sync():
            msg_data = service.users().messages().get(
                userId="me",
                id=message_id,
                format="full",
            ).execute()
            return parse_message_payload(msg_data)

        email_data = await asyncio.to_thread(_get_sync)

        return (
            f"### Subject: {email_data['subject']}\n"
            f"- **Message ID**: `{email_data['id']}`\n"
            f"- **From**: {email_data['from']}\n"
            f"- **To**: {email_data['to']}\n"
            f"- **Date**: {email_data['date']}\n\n"
            f"#### Content:\n{email_data['body']}"
        )

    except Exception as exc:
        logger.exception("Error executing get_email_details for message %s: %s", message_id, exc)
        return f"Failed to retrieve email details: {str(exc)}"


@tool
async def send_email(
    to: str,
    subject: str,
    body: str,
    config: Optional[RunnableConfig] = None,
) -> str:
    """
    Send an email from the user's Gmail account via the Google Gmail API.

    Args:
        to: Recipient email address (e.g. 'recipient@example.com').
        subject: The subject line of the email.
        body: The plain text body content of the email.
        config: Execution runtime config containing user authentication info.

    Returns:
        Confirmation status and sent message ID.
    """
    service, err = await get_authenticated_gmail_service(config)
    if err:
        return err

    try:
        def _send_sync():
            message = EmailMessage()
            message.set_content(body)
            message["To"] = to
            message["Subject"] = subject
            raw_data = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
            sent_msg = service.users().messages().send(userId="me", body={"raw": raw_data}).execute()
            return sent_msg

        res = await asyncio.to_thread(_send_sync)
        return f"Email successfully sent to `{to}` (Message ID: `{res.get('id')}`)."

    except Exception as exc:
        logger.exception("Error executing send_email: %s", exc)
        return f"Failed to send email: {str(exc)}"


class FetchUserEmailsTool(Tools):
    cls_tool = fetch_user_emails


class GetEmailDetailsTool(Tools):
    cls_tool = get_email_details


class SendEmailTool(Tools):
    cls_tool = send_email
