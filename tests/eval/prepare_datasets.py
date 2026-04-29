"""
Run once to build tests/eval/datasets/golden_transcripts.json from:
  - AMI Meeting Corpus   (HuggingFace: edinburghcristin/ami-corpus)
  - MeetingBank          (HuggingFace: huuuyeah/meetingbank)
  - 2 hand-labeled tech meeting examples

Usage:
    python tests/eval/prepare_datasets.py
"""
import json
import os

from datasets import load_dataset

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from app.core.config import get_settings

_settings = get_settings()
OUTPUT = _settings.eval_fixtures_path
fixtures = []


def load_ami():
    print("Loading AMI corpus...")
    try:
        ds = load_dataset(_settings.ami_dataset_repo, split="test", trust_remote_code=True)
    except Exception as e:
        print(f"  AMI load failed ({e}), skipping.")
        return
    for row in ds.select(range(min(5, len(ds)))):
        transcript = row.get("transcript") or row.get("text") or ""
        if not transcript:
            continue
        action_items_raw = row.get("action_items") or []
        decisions_raw = row.get("decisions") or []
        fixtures.append({
            "id": f"ami_{row.get('meeting_id', len(fixtures))}",
            "source": "AMI",
            "transcript": transcript[:4000],
            "expected_signals": [
                {"signal_type": "decision", "summary": d}
                for d in (decisions_raw if isinstance(decisions_raw, list) else [decisions_raw])
                if d
            ],
            "expected_action_items": [
                {"task": a if isinstance(a, str) else str(a), "owner_name": None,
                 "deadline": None, "supporting_quote": ""}
                for a in (action_items_raw if isinstance(action_items_raw, list) else [action_items_raw])
                if a
            ],
        })
    print(f"  Loaded {len([f for f in fixtures if f['source']=='AMI'])} AMI examples.")


def load_meetingbank():
    print("Loading MeetingBank...")
    try:
        ds = load_dataset(_settings.meetingbank_dataset_repo, split="test", trust_remote_code=True)
    except Exception as e:
        print(f"  MeetingBank load failed ({e}), skipping.")
        return
    count_before = len(fixtures)
    for row in ds.select(range(min(5, len(ds)))):
        transcript = row.get("transcript") or ""
        if not transcript:
            continue
        fixtures.append({
            "id": f"meetingbank_{len(fixtures)}",
            "source": "MeetingBank",
            "transcript": transcript[:4000],
            "expected_signals": [],
            "expected_action_items": [],
            "summary_reference": row.get("summary") or "",
        })
    print(f"  Loaded {len(fixtures) - count_before} MeetingBank examples.")


HAND_LABELED = [
    {
        "id": "hand_1",
        "source": "hand_labeled",
        "transcript": (
            "Alice: We've decided to migrate to PostgreSQL by end of Q1.\n"
            "Bob: I'll handle the schema migration. Target is March 15th.\n"
            "Alice: One blocker — we don't have staging credentials yet.\n"
            "Bob: I'll get those from DevOps by Monday.\n"
            "Carol: We need to update the API docs before the client demo on Wednesday."
        ),
        "expected_signals": [
            {"signal_type": "decision", "summary": "Migrate to PostgreSQL by end of Q1"},
            {"signal_type": "blocker", "summary": "No staging credentials"},
            {"signal_type": "commitment", "summary": "Bob will get staging credentials by Monday"},
        ],
        "expected_action_items": [
            {"task": "Handle the schema migration", "owner_name": "Bob", "deadline": "March 15th",
             "supporting_quote": "I'll handle the schema migration. Target is March 15th"},
            {"task": "Get staging credentials from DevOps", "owner_name": "Bob", "deadline": "Monday",
             "supporting_quote": "I'll get those from DevOps by Monday"},
            {"task": "Update the API docs before client demo", "owner_name": "Carol", "deadline": "Wednesday",
             "supporting_quote": "we need to update the API docs before the client demo on Wednesday"},
        ],
    },
    {
        "id": "hand_2",
        "source": "hand_labeled",
        "transcript": (
            "Dave: The API latency is above 500ms in production. That's a blocker for the release.\n"
            "Eve: I can profile the slow queries this week and send a report.\n"
            "Dave: We also committed to delivering the dashboard to the client by Thursday.\n"
            "Eve: Got it. I'll prioritize the dashboard over the profiling."
        ),
        "expected_signals": [
            {"signal_type": "blocker", "summary": "API latency above 500ms blocking release"},
            {"signal_type": "commitment", "summary": "Deliver dashboard to client by Thursday"},
        ],
        "expected_action_items": [
            {"task": "Profile slow queries and send report", "owner_name": "Eve", "deadline": "this week",
             "supporting_quote": "I can profile the slow queries this week and send a report"},
            {"task": "Prioritize dashboard delivery", "owner_name": "Eve", "deadline": "Thursday",
             "supporting_quote": "I'll prioritize the dashboard over the profiling"},
        ],
    },
]


if __name__ == "__main__":
    load_ami()
    load_meetingbank()
    fixtures.extend(HAND_LABELED)
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(fixtures, f, indent=2)
    print(f"\nWrote {len(fixtures)} fixtures to {OUTPUT}")
