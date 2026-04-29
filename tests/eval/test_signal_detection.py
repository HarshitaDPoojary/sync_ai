"""
Evaluates signal detection (decision/blocker/commitment) using LangSmith evaluate().
Requires: LANGCHAIN_API_KEY set and LANGCHAIN_TRACING_V2=true.
Run prepare_datasets.py first if you want AMI/MeetingBank examples.

Results visible in LangSmith dashboard at smith.langchain.com.
"""
import json
import os
import pytest

from app.core.config import get_settings

_settings = get_settings()

with open(_settings.eval_fixtures_path) as f:
    GOLDEN = [g for g in json.load(f) if g.get("expected_signals")]

DATASET_NAME = _settings.langsmith_signal_dataset


def _upload_dataset_if_needed():
    from langsmith import Client
    client = Client()
    existing = [d.name for d in client.list_datasets()]
    if DATASET_NAME not in existing:
        examples = [
            {"inputs": {"transcript": g["transcript"]},
             "outputs": {"signals": g["expected_signals"]}}
            for g in GOLDEN
        ]
        client.create_dataset(DATASET_NAME, description="Signal detection eval from AMI + hand-labeled")
        client.create_examples(
            inputs=[e["inputs"] for e in examples],
            outputs=[e["outputs"] for e in examples],
            dataset_name=DATASET_NAME,
        )


def run_signal_detection(inputs: dict) -> dict:
    from app.agents.analysis import run_analysis_node
    from app.core.graph import make_initial_state, TranscriptChunk
    state = make_initial_state("eval", [])
    state["analysis_window"] = [
        TranscriptChunk(speaker="Test", text=inputs["transcript"], timestamp=0.0, sequence_num=1)
    ]
    state["token_count"] = 600
    state["last_analysis_token_count"] = 0
    result_state = run_analysis_node(state)
    return {"signals": result_state["signals"]}


class SignalF1Evaluator:
    def __call__(self, run, example) -> dict:
        predicted = {s["signal_type"] for s in (run.outputs or {}).get("signals", [])}
        expected = {s["signal_type"] for s in (example.outputs or {}).get("signals", [])}
        tp = len(predicted & expected)
        precision = tp / len(predicted) if predicted else 0
        recall = tp / len(expected) if expected else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return {"key": "signal_f1", "score": f1}


@pytest.mark.skipif(not os.getenv("LANGCHAIN_API_KEY"), reason="LANGCHAIN_API_KEY not set")
def test_signal_detection_f1():
    from langsmith import evaluate
    _upload_dataset_if_needed()
    results = evaluate(
        run_signal_detection,
        data=DATASET_NAME,
        evaluators=[SignalF1Evaluator()],
        experiment_prefix="signal-detection",
    )
    scores = [r.get("evaluation_results", {}).get("results", [{}])[0].get("score", 0) for r in results]
    avg_f1 = sum(scores) / max(len(scores), 1)
    assert avg_f1 >= 0.8, f"Average signal detection F1={avg_f1:.2f} below 0.80"
