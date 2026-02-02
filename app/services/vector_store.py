"""Vector store service using ChromaDB for RAG."""

import os
from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings

settings = get_settings()


class VectorStore:
    """ChromaDB-based vector store for RAG retrieval."""

    def __init__(self, persist_directory: str = "./chroma_data"):
        """
        Initialize vector store.

        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Default embedding function (uses sentence-transformers locally)
        # For production, could switch to OpenAI embeddings
        self.embedding_function = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()

    def get_or_create_collection(self, content_id: int) -> chromadb.Collection:
        """
        Get or create a collection for a specific content item.

        Args:
            content_id: Content ID to use as collection name

        Returns:
            ChromaDB collection
        """
        collection_name = f"content_{content_id}"
        return self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        content_id: int,
        chunks: list[dict],
        use_english: bool = True,
    ) -> list[str]:
        """
        Add chunks to the vector store.

        Args:
            content_id: Content ID
            chunks: List of chunk dictionaries with text, text_english, etc.
            use_english: Whether to embed English text (better for multilingual retrieval)

        Returns:
            List of embedding IDs
        """
        collection = self.get_or_create_collection(content_id)

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            chunk_id = f"chunk_{content_id}_{chunk['index']}"
            ids.append(chunk_id)

            # Use English text for embedding if available (better cross-lingual retrieval)
            text_to_embed = chunk.get("text_english") or chunk.get("text", "")
            documents.append(text_to_embed)

            metadatas.append({
                "content_id": content_id,
                "chunk_index": chunk["index"],
                "original_text": chunk.get("text", ""),
                "start_time": chunk.get("start_time") or 0,
                "end_time": chunk.get("end_time") or 0,
                "token_count": chunk.get("token_count", 0),
            })

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        return ids

    def search(
        self,
        content_id: int,
        query: str,
        n_results: int = 5,
        min_score: float = 0.3,
    ) -> list[dict]:
        """
        Search for relevant chunks.

        Args:
            content_id: Content ID to search within
            query: Search query (will be embedded)
            n_results: Maximum number of results
            min_score: Minimum similarity score (0-1, higher = more similar)

        Returns:
            List of chunk dictionaries with similarity scores
        """
        collection = self.get_or_create_collection(content_id)

        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances; convert to similarity score
                # For cosine distance: similarity = 1 - distance
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 - distance

                if similarity >= min_score:
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    chunks.append({
                        "id": chunk_id,
                        "chunk_index": metadata.get("chunk_index", 0),
                        "text": metadata.get("original_text", ""),
                        "text_embedded": results["documents"][0][i] if results["documents"] else "",
                        "start_time": metadata.get("start_time"),
                        "end_time": metadata.get("end_time"),
                        "token_count": metadata.get("token_count", 0),
                        "similarity": round(similarity, 3),
                    })

        return chunks

    def delete_collection(self, content_id: int):
        """
        Delete a collection for a content item.

        Args:
            content_id: Content ID
        """
        collection_name = f"content_{content_id}"
        try:
            self.client.delete_collection(collection_name)
        except ValueError:
            pass  # Collection doesn't exist

    def get_collection_stats(self, content_id: int) -> dict:
        """
        Get statistics for a collection.

        Args:
            content_id: Content ID

        Returns:
            Dictionary with collection stats
        """
        collection = self.get_or_create_collection(content_id)
        return {
            "name": collection.name,
            "count": collection.count(),
        }
