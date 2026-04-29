from typing import Any, Dict, List

from slack_sdk import WebClient

from app.core.config import get_settings


class SlackClient:
    def __init__(self, bot_token: str | None = None, channel_id: str | None = None):
        settings = get_settings()
        token = bot_token or settings.slack_bot_token
        self._channel = channel_id or settings.slack_channel_id
        self._sync_client = WebClient(token=token)

    def _format_action_items(self, meeting_title: str, items: List[Dict[str, Any]]) -> str:
        lines = [f"*Action Items from: {meeting_title}*\n"]
        for item in items:
            owner = item.get("owner_name") or "Unassigned"
            deadline = item.get("deadline") or "No deadline"
            flag = " ⚠️ _needs review_" if item.get("needs_review") else ""
            lines.append(f"• *{item['task']}* — {owner} by {deadline}{flag}")
        return "\n".join(lines)

    def _format_summary(
        self,
        meeting_title: str,
        decisions: List[str],
        blockers: List[str],
        next_steps: List[str],
    ) -> str:
        sections = [f"*Meeting Summary: {meeting_title}*\n"]
        if decisions:
            sections.append("*Decisions:*\n" + "\n".join(f"• {d}" for d in decisions))
        if blockers:
            sections.append("*Blockers:*\n" + "\n".join(f"• {b}" for b in blockers))
        if next_steps:
            sections.append("*Next Steps:*\n" + "\n".join(f"• {s}" for s in next_steps))
        return "\n\n".join(sections)

    def send_action_items_sync(self, meeting_title: str, items: List[Dict[str, Any]]) -> None:
        text = self._format_action_items(meeting_title, items)
        self._sync_client.chat_postMessage(channel=self._channel, text=text)

    def send_summary_sync(
        self,
        meeting_title: str,
        decisions: List[str],
        blockers: List[str],
        next_steps: List[str],
    ) -> None:
        text = self._format_summary(meeting_title, decisions, blockers, next_steps)
        self._sync_client.chat_postMessage(channel=self._channel, text=text)
