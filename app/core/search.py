from typing import Any, Dict, List

from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from app.core.config import get_settings

_search_chain_cache = None


def get_search_chain() -> RetrievalQA:
    global _search_chain_cache
    if _search_chain_cache is None:
        settings = get_settings()
        embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        vectorstore = Chroma(
            collection_name="transcripts",
            embedding_function=embeddings,
            persist_directory=settings.chroma_persist_dir,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        llm = ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
        )
        _search_chain_cache = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
        )
    return _search_chain_cache


def semantic_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    chain = get_search_chain()
    result = chain.invoke({"query": query})
    docs = result.get("source_documents", [])
    return [
        {
            "text": doc.page_content,
            "meeting_id": doc.metadata.get("meeting_id", ""),
            "sequence": doc.metadata.get("sequence", 0),
        }
        for doc in docs[:limit]
    ]
