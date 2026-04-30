from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from app.core.config import get_settings


def _get_vectorstore(user_id: Optional[str] = None) -> Chroma:
    settings = get_settings()
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=settings.huggingface_api_key,
        model_name=settings.embedding_model,
    )
    collection_name = f"user_{user_id}" if user_id else "transcripts"
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )


def semantic_search(query: str, user_id: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    vectorstore = _get_vectorstore(user_id)
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
    user_id: Optional[str] = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Direct ChromaDB similarity search. Used by eval tests."""
    return semantic_search(query, user_id=user_id, limit=limit)
