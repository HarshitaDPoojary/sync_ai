import asyncio
import functools
import threading
from functools import lru_cache
from typing import Any, Dict, List, Optional

import chromadb
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from app.agents.analysis import run_analysis_node
from app.agents.delivery import run_delivery_node
from app.agents.extraction import run_extraction_node
from app.agents.storage import run_storage_node
from app.core.config import get_settings
from app.core.graph import MeetingState, TranscriptChunk, build_graph, make_initial_state, should_analyze
from app.core.recall import RecallClient
from app.models.db import get_engine


@lru_cache(maxsize=1)
def _get_chroma_client(persist_dir: str) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=persist_dir)


class MeetingSession:
    def __init__(
        self,
        meeting_id: str,
        meeting_url: str,
        title: str,
        participant_emails: List[str],
        recall_api_key: Optional[str] = None,
        webhook_base_url: Optional[str] = None,
        slack_channel_id: Optional[str] = None,
        slack_bot_token: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        settings = get_settings()
        self.meeting_id = meeting_id
        self.meeting_url = meeting_url
        self.title = title
        self.participant_emails = participant_emails
        self.bot_id: Optional[str] = None
        self.user_id = user_id

        self._recall = RecallClient(api_key=recall_api_key)
        self._webhook_base_url = webhook_base_url or settings.webhook_base_url
        self._slack_channel_id = slack_channel_id
        self._slack_bot_token = slack_bot_token
        self._engine = get_engine()
        self._lock = threading.Lock()

        chroma_client = _get_chroma_client(settings.chroma_persist_dir)
        collection_name = f"user_{user_id}" if user_id else "transcripts"
        self._chroma = chroma_client.get_or_create_collection(collection_name)
        self._embedder = HuggingFaceInferenceAPIEmbeddings(
            api_key=settings.huggingface_api_key,
            model_name=settings.embedding_model,
        )

        participants = [{"name": e.split("@")[0], "email": e} for e in participant_emails]
        self._state: MeetingState = make_initial_state(meeting_id, participants)

        self._graph = build_graph(
            analysis_node=run_analysis_node,
            extraction_node=run_extraction_node,
            delivery_node=functools.partial(
                run_delivery_node,
                meeting_title=title,
                slack_channel_id=slack_channel_id,
                slack_bot_token=slack_bot_token,
            ),
            storage_node=functools.partial(
                run_storage_node,
                engine=self._engine,
                chroma_collection=self._chroma,
                embedder=self._embedder,
            ),
        )

    async def start(self) -> None:
        webhook_url = f"{self._webhook_base_url}/webhook/recall/{self.meeting_id}"
        bot = await self._recall.create_bot_with_webhook(
            meeting_url=self.meeting_url,
            webhook_url=webhook_url,
        )
        self.bot_id = bot["id"]

    async def stop(self) -> None:
        if self.bot_id:
            try:
                await self._recall.remove_bot(self.bot_id)
            except Exception:
                pass
        with self._lock:
            self._state["meeting_ended"] = True
        await asyncio.get_running_loop().run_in_executor(None, self._run_graph)

    def ingest_chunk(self, speaker: str, text: str, timestamp: float) -> None:
        with self._lock:
            seq = len(self._state["transcript_chunks"]) + 1
            chunk = TranscriptChunk(speaker=speaker, text=text, timestamp=timestamp, sequence_num=seq)

            settings = get_settings()
            tokens = len(text.split())
            cutoff_time = timestamp - settings.analysis_window_seconds
            window = [
                c for c in self._state["transcript_chunks"] if c["timestamp"] >= cutoff_time
            ] + [chunk]

            chunks = self._state["transcript_chunks"]
            chunks.append(chunk)
            self._state = {
                **self._state,
                "transcript_chunks": chunks,
                "analysis_window": window,
                "token_count": self._state["token_count"] + tokens,
            }

            if should_analyze(self._state):
                self._run_graph()

    def _run_graph(self) -> None:
        result = self._graph.invoke(self._state)
        self._state = result

    def get_state(self) -> MeetingState:
        return dict(self._state)