"""
Evaluates action item extraction precision and hallucination rate using LangSmith evaluate().
Requires: LANGCHAIN_API_KEY set, GROQ_API_KEY set.
"""
import json
import os
import pytest

with open("tests/eval/datasets/golden_transcripts.json") as f:
    GOLDEN = [g for g in json.load(f) if g.get("expected_action_items") and g["source"] == "hand_labeled"]

DATASET_NAME = "sync_ai-action-extraction"
PARTICIPANTS = [{"name": n, "email": f"{n.lower()}@test.com"}
                for n in ["Alice", "Bob", "Carol", "Dave", "Eve"]]


def _upload_dataset_if_needed():
    from langsmith import Client
    client = Client()
    existing = [d.name for d in client.list_datasets()]
    if DATASET_NAME not in existing:
        examples = [
            {"inputs": {"transcript": g["transcript"], "participants": PARTICIPANTS},
             "outputs": {"action_items": g["expected_action_items"]}}
            for g in GOLDEN
        ]
        client.create_dataset(DATASET_NAME, description="Action extraction eval — hand labeled")
        client.create_examples(
            inputs=[e["inputs"] for e in examples],
            outputs=[e["outputs"] for e in examples],
            dataset_name=DATASET_NAME,
        )


def run_extraction(inputs: dict) -> dict:
    from app.agents.extraction import run_extraction_node
    from app.core.graph import make_initial_state, TranscriptChunk
    transcript = inputs["transcript"]
    lines = transcript.split("\n")
    chunks = []
    for i, line in enumerate(lines):
        if ": " in line:
            speaker, text = line.split(": ", 1)
        else:
            speaker, text = "Speaker", line
        chunks.append(TranscriptChunk(speaker=speaker, text=text, timestamp=float(i), sequence_num=i + 1))
    state = make_initial_state("eval", inputs.get("participants", []))
    state["transcript_chunks"] = chunks
    state["meeting_ended"] = True
    result_state = run_extraction_node(state)
    return {"action_items": result_state["action_items"]}


class ExtractionPrecisionEvaluator:
    def __call__(self, run, example) -> dict:
        predicted_tasks = {i["task"] for i in (run.outputs or {}).get("action_items", [])}
        expected_tasks = {i["task"] for i in (example.outputs or {}).get("action_items", [])}
        tp = len(predicted_tasks & expected_tasks)
        precision = tp / len(predicted_tasks) if predicted_tasks else 0
        return {"key": "extraction_precision", "score": precision}


class HallucinationQualityEvaluator:
    def __call__(self, run, example) -> dict:
        items = (run.outputs or {}).get("action_items", [])
        hallucinated = [i for i in items if i.get("needs_review")]
        rate = len(hallucinated) / len(items) if items else 0
        return {"key": "hallucination_quality", "score": 1 - rate}


@pytest.mark.skipif(not os.getenv("LANGCHAIN_API_KEY"), reason="LANGCHAIN_API_KEY not set")
def test_action_extraction_quality():
    from langsmith import evaluate
    _upload_dataset_if_needed()
    results = evaluate(
        run_extraction,
        data=DATASET_NAME,
        evaluators=[ExtractionPrecisionEvaluator(), HallucinationQualityEvaluator()],
        experiment_prefix="action-extraction",
    )
    precisions = []
    hallucination_scores = []
    for r in results:
        for res in r.get("evaluation_results", {}).get("results", []):
            if res.get("key") == "extraction_precision":
                precisions.append(res.get("score", 0))
            elif res.get("key") == "hallucination_quality":
                hallucination_scores.append(res.get("score", 0))

    if precisions:
        avg_precision = sum(precisions) / len(precisions)
        assert avg_precision >= 0.8, f"Average extraction precision={avg_precision:.2f} below 0.80"
    if hallucination_scores:
        avg_quality = sum(hallucination_scores) / len(hallucination_scores)
        assert avg_quality >= 0.95, f"Hallucination quality={avg_quality:.2f} below 0.95"
