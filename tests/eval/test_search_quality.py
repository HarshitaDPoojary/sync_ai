"""
Evaluates semantic search quality using direct ChromaDB similarity search.
No LangSmith API key required — runs offline against mocked ChromaDB.
"""
import pytest
from unittest.mock import MagicMock, patch

SEARCH_CASES = [
    {
        "query": "PostgreSQL migration deadline",
        "expected_keywords": ["PostgreSQL", "migration", "March"],
        "corpus": "Alice: We decided to migrate to PostgreSQL by end of Q1. Bob: Schema migration target is March 15th.",
    },
    {
        "query": "API latency production blocker",
        "expected_keywords": ["latency", "500ms", "blocker"],
        "corpus": "Dave: The API latency is above 500ms in production. That's a blocker for the release.",
    },
    {
        "query": "dashboard client delivery Thursday",
        "expected_keywords": ["dashboard", "client", "Thursday"],
        "corpus": "Dave: We committed to delivering the dashboard to the client by Thursday. Eve: I'll prioritize the dashboard.",
    },
]


@pytest.mark.parametrize("case", SEARCH_CASES, ids=[c["query"][:30] for c in SEARCH_CASES])
def test_search_returns_relevant_chunk(case):
    mock_doc = MagicMock()
    mock_doc.page_content = case["corpus"]
    mock_doc.metadata = {"meeting_id": "eval", "sequence": 0}

    with patch("app.core.search.Chroma") as MockChroma, \
         patch("app.core.search.HuggingFaceInferenceAPIEmbeddings"):
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [mock_doc]
        MockChroma.return_value = mock_vs

        from app.core.search import search_transcripts
        results = search_transcripts(case["query"], user_id=None, limit=3)

    assert len(results) > 0, f"No results for: {case['query']}"
    result_text = results[0]["text"].lower()
    matched = [kw for kw in case["expected_keywords"] if kw.lower() in result_text]
    match_rate = len(matched) / len(case["expected_keywords"])
    assert match_rate >= 0.6, (
        f"Query '{case['query']}': matched {len(matched)}/{len(case['expected_keywords'])} "
        f"keywords {matched} in: {result_text[:100]}"
    )
