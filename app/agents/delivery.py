import logging
from collections import defaultdict
from typing import Optional

from app.core.graph import MeetingState
from app.integrations.slack import SlackClient

logger = logging.getLogger("sync_ai.delivery")


def _get_slack_client(
    channel_id: Optional[str],
    bot_token: Optional[str],
) -> Optional[SlackClient]:
    if not bot_token or not channel_id:
        return None
    return SlackClient(bot_token=bot_token, channel_id=channel_id)


def _send_gmail(
    items: list,
    meeting_title: str,
    gmail_integration=None,
) -> None:
    """Send each owner their own action items by email using per-user OAuth credentials."""
    if gmail_integration is None:
        return
    from app.core.config import get_settings
    from app.integrations.gmail import GmailClient
    settings = get_settings()
    client = GmailClient(
        access_token=gmail_integration.access_token,
        refresh_token=gmail_integration.refresh_token,
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
    )
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
    slack_bot_token: Optional[str] = None,
    gmail_integration=None,
) -> MeetingState:
    items = [dict(item) for item in state["action_items"]]
    summary = state.get("summary") or {}

    slack = _get_slack_client(slack_channel_id, slack_bot_token)
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
        _send_gmail(items, meeting_title, gmail_integration)

    return state
