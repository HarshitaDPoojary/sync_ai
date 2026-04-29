import logging
from collections import defaultdict
from typing import Optional

from app.core.config import get_settings
from app.core.graph import MeetingState
from app.integrations.gmail import GmailClient
from app.integrations.slack import SlackClient

logger = logging.getLogger("sync_ai.delivery")


def _get_slack_client(channel_id: Optional[str]) -> Optional[SlackClient]:
    settings = get_settings()
    token = settings.slack_bot_token
    channel = channel_id or settings.slack_channel_id
    if not token or not channel:
        return None
    return SlackClient(bot_token=token, channel_id=channel)


def _send_gmail(items: list, meeting_title: str) -> None:
    """Send each owner their own action items by email."""
    settings = get_settings()
    if not settings.gmail_credentials_json:
        return
    client = GmailClient()
    by_owner: dict = defaultdict(list)
    for item in items:
        email = item.get("owner_email")
        if email:
            by_owner[email].append(item)
    for email, owner_items in by_owner.items():
        owner_name = owner_items[0].get("owner_name") or email.split("@")[0]
        try:
            client.send_action_items(email, owner_name, meeting_title, owner_items)
            logger.info("gmail_sent to=%s count=%d", email, len(owner_items))
        except Exception as exc:
            logger.warning("gmail_failed to=%s error=%s", email, exc)


def run_delivery_node(
    state: MeetingState,
    meeting_title: str = "Meeting",
    slack_channel_id: Optional[str] = None,
) -> MeetingState:
    items = [dict(item) for item in state["action_items"]]
    summary = state.get("summary") or {}

    slack = _get_slack_client(slack_channel_id)
    if slack and items:
        try:
            slack.send_action_items_sync(meeting_title, items)
        except Exception as exc:
            logger.warning("slack_action_items_failed error=%s", exc)

    if slack and summary:
        try:
            slack.send_summary_sync(
                meeting_title,
                summary.get("decisions", []),
                summary.get("blockers", []),
                summary.get("next_steps", []),
            )
        except Exception as exc:
            logger.warning("slack_summary_failed error=%s", exc)

    if items:
        _send_gmail(items, meeting_title)

    return state
