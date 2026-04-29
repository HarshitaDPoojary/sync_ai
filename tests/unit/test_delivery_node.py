from unittest.mock import MagicMock, patch
from app.agents.delivery import run_delivery_node
from app.core.graph import make_initial_state, ExtractedActionItem


def test_delivery_node_calls_slack():
    state = make_initial_state("m1", [])
    state["action_items"] = [
        ExtractedActionItem(
            task="Deploy API", owner_name="Bob", owner_email="bob@x.com",
            deadline="Friday", confidence=0.9, supporting_quote="Bob will deploy",
            needs_review=False,
        )
    ]
    state["summary"] = {"decisions": ["Use PG"], "blockers": [], "commitments": [], "next_steps": []}

    mock_slack = MagicMock()
    mock_slack.send_action_items_sync = MagicMock()
    mock_slack.send_summary_sync = MagicMock()

    with patch("app.agents.delivery.get_slack_client", return_value=mock_slack):
        run_delivery_node(state, meeting_title="Sprint Review")

    mock_slack.send_action_items_sync.assert_called_once()
    mock_slack.send_summary_sync.assert_called_once()


def test_delivery_node_skips_when_no_slack():
    state = make_initial_state("m1", [])
    state["action_items"] = []
    state["summary"] = {}
    with patch("app.agents.delivery.get_slack_client", return_value=None):
        result = run_delivery_node(state)
    assert result["action_items"] == []
