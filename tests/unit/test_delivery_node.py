from unittest.mock import MagicMock, call, patch
from app.agents.delivery import run_delivery_node, _send_gmail
from app.core.graph import make_initial_state, ExtractedActionItem


def _state_with_item(owner_email="bob@x.com"):
    state = make_initial_state("m1", [])
    state["action_items"] = [
        ExtractedActionItem(
            task="Deploy API", owner_name="Bob", owner_email=owner_email,
            deadline="Friday", confidence=0.9, supporting_quote="Bob will deploy",
            needs_review=False,
        )
    ]
    state["summary"] = {"decisions": ["Use PG"], "blockers": [], "commitments": [], "next_steps": []}
    return state


def test_delivery_node_calls_slack():
    state = _state_with_item()
    mock_slack = MagicMock()

    with patch("app.agents.delivery._get_slack_client", return_value=mock_slack), \
         patch("app.agents.delivery._send_gmail"):
        run_delivery_node(state, meeting_title="Sprint Review", slack_channel_id="C123")

    mock_slack.send_action_items_sync.assert_called_once()
    mock_slack.send_summary_sync.assert_called_once()


def test_delivery_node_skips_when_no_slack():
    state = make_initial_state("m1", [])
    state["action_items"] = []
    state["summary"] = {}
    with patch("app.agents.delivery._get_slack_client", return_value=None), \
         patch("app.agents.delivery._send_gmail"):
        result = run_delivery_node(state)
    assert result["action_items"] == []


def test_delivery_node_calls_gmail_per_owner():
    state = _state_with_item(owner_email="bob@x.com")
    mock_gmail = MagicMock()

    with patch("app.agents.delivery._get_slack_client", return_value=None), \
         patch("app.agents.delivery.GmailClient", return_value=mock_gmail), \
         patch("app.agents.delivery.get_settings") as mock_settings:
        mock_settings.return_value.gmail_credentials_json = "dummytoken"
        run_delivery_node(state, meeting_title="Sprint Review")

    mock_gmail.send_action_items.assert_called_once()
    args = mock_gmail.send_action_items.call_args
    assert args[0][0] == "bob@x.com"
    assert args[0][2] == "Sprint Review"


def test_delivery_node_skips_gmail_when_no_credentials():
    state = _state_with_item()
    mock_gmail = MagicMock()

    with patch("app.agents.delivery._get_slack_client", return_value=None), \
         patch("app.agents.delivery.GmailClient", return_value=mock_gmail), \
         patch("app.agents.delivery.get_settings") as mock_settings:
        mock_settings.return_value.gmail_credentials_json = ""
        run_delivery_node(state, meeting_title="Sprint Review")

    mock_gmail.send_action_items.assert_not_called()
