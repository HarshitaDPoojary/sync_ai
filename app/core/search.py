from functools import lru_cache
from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from app.core.config import get_settings


@lru_cache(maxsize=1)
def _get_vectorstore(embedding_model: str, persist_dir: str) -> Chroma:
    settings = get_settings()
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=settings.huggingface_api_key,
        model_name=embedding_model,
    )
    return Chroma(
        collection_name="transcripts",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


def semantic_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    settings = get_settings()
    vectorstore = _get_vectorstore(settings.embedding_model, settings.chroma_persist_dir)
    docs = vectorstore.similarity_search(query, k=limit)
    return [
        {
            "text": doc.page_content,
            "meeting_id": doc.metadata.get("meeting_id", ""),
            "sequence": doc.metadata.get("sequence", 0),
        }
        for doc in docs
    ]


def search_transcripts(
    query: str,
    chroma_persist_dir: str | None = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Direct ChromaDB similarity search. Used by eval tests."""
    settings = get_settings()
    persist_dir = chroma_persist_dir or settings.chroma_persist_dir
    vectorstore = _get_vectorstore(settings.embedding_model, persist_dir)
    docs = vectorstore.similarity_search(query, k=limit)
    return [
        {
            "text": doc.page_content,
            "meeting_id": doc.metadata.get("meeting_id", ""),
            "sequence": doc.metadata.get("sequence", 0),
        }
        for doc in docs
    ]
