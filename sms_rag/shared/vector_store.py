"""Abstract interface for vector store operations."""

from abc import ABC, abstractmethod
from typing import Any

from sms_rag.shared.models import SearchResult


class VectorStoreInterface(ABC):
    """Abstract base class for vector store implementations.

    Enables swapping between ChromaDB, Qdrant, Pinecone, or other providers
    without modifying consuming code.
    """

    @abstractmethod
    def store_embeddings(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Store embeddings with associated metadata.

        Args:
            ids: Unique identifiers for each embedding.
            embeddings: Vector representations of the documents.
            documents: Original text content of each document.
            metadatas: Metadata dictionaries for each document.
        """
        ...

    @abstractmethod
    def query_by_similarity(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Query for similar documents with optional metadata filters.

        Args:
            query_embedding: The query vector to search against.
            top_k: Maximum number of results to return.
            metadata_filters: Optional filters to narrow the search.

        Returns:
            List of SearchResult objects sorted by relevance score descending.
        """
        ...

    @abstractmethod
    def delete_by_ids(self, ids: list[str]) -> None:
        """Delete embeddings by their identifiers.

        Args:
            ids: List of embedding identifiers to delete.
        """
        ...

    @abstractmethod
    def has_document(self, source_filename: str) -> bool:
        """Check if a source file has already been indexed.

        Args:
            source_filename: The filename to check for existing embeddings.

        Returns:
            True if the source file has been indexed, False otherwise.
        """
        ...
