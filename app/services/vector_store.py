"""ChromaDB vector store: one collection per processed video."""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Embeds transcript chunks and retrieves the most similar ones for a query."""

    def __init__(self, persist_directory: str | None = None):
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        from chromadb.utils import embedding_functions

        self.persist_directory = persist_directory or get_settings().chroma_dir
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # Local sentence-transformer model: no API key needed to embed or search.
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

    @staticmethod
    def collection_name(content_id: int) -> str:
        return f"content_{content_id}"

    def get_or_create_collection(self, content_id: int):
        return self.client.get_or_create_collection(
            name=self.collection_name(content_id),
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, content_id: int, chunks: list[dict]) -> list[str]:
        """Index chunks. English text is embedded when available for cross-lingual retrieval."""
        if not chunks:
            return []
        collection = self.get_or_create_collection(content_id)

        ids, documents, metadatas = [], [], []
        for chunk in chunks:
            ids.append(f"chunk_{content_id}_{chunk['index']}")
            documents.append(chunk.get("text_english") or chunk.get("text", ""))
            metadata = {
                "content_id": content_id,
                "chunk_index": chunk["index"],
                "original_text": chunk.get("text", ""),
                "token_count": chunk.get("token_count") or 0,
            }
            # Chroma metadata cannot hold None; omit unknown timestamps rather than faking 0:00.
            for key in ("start_time", "end_time"):
                if chunk.get(key) is not None:
                    metadata[key] = chunk[key]
            metadatas.append(metadata)

        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        return ids

    def search(
        self,
        content_id: int,
        query: str,
        n_results: int = 5,
        min_score: float = 0.3,
    ) -> list[dict]:
        """Return chunks with cosine similarity >= ``min_score``, best first."""
        collection = self.get_or_create_collection(content_id)
        if collection.count() == 0:
            return []

        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for chunk_id, document, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            strict=True,
        ):
            similarity = 1 - distance  # cosine distance -> similarity
            if similarity < min_score:
                continue
            chunks.append(
                {
                    "id": chunk_id,
                    "chunk_index": metadata.get("chunk_index", 0),
                    "text": metadata.get("original_text", ""),
                    "text_embedded": document,
                    "start_time": metadata.get("start_time"),
                    "end_time": metadata.get("end_time"),
                    "token_count": metadata.get("token_count", 0),
                    "similarity": round(similarity, 3),
                }
            )
        return chunks

    def delete_collection(self, content_id: int) -> None:
        try:
            self.client.delete_collection(self.collection_name(content_id))
        except Exception:  # Chroma raises different types across versions when missing
            logger.debug("No vector collection to delete for content %s", content_id)
