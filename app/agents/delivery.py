from typing import Optional

from app.core.config import get_settings
from app.core.graph import MeetingState
from app.integrations.slack import SlackClient

_slack_client: Optional[SlackClient] = None


def get_slack_client() -> Optional[SlackClient]:
    global _slack_client
    settings = get_settings()
    if settings.slack_bot_token and settings.slack_channel_id and _slack_client is None:
        _slack_client = SlackClient()
    return _slack_client


def run_delivery_node(state: MeetingState, meeting_title: str = "Meeting") -> MeetingState:
    slack = get_slack_client()
    items = [dict(item) for item in state["action_items"]]
    summary = state.get("summary") or {}

    if slack and items:
        slack.send_action_items_sync(meeting_title, items)

    if slack and summary:
        slack.send_summary_sync(
            meeting_title,
            summary.get("decisions", []),
            summary.get("blockers", []),
            summary.get("next_steps", []),
        )

    return state
