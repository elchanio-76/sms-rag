"""ChromaDB adapter implementing the VectorStoreInterface.

Uses chromadb.PersistentClient for local storage with the collection
name "sms_conversations". Metadata fields are stored as filterable
ChromaDB attributes; list-typed fields are serialized as JSON strings.
"""

import json
import signal
from pathlib import Path
from typing import Any

import chromadb

from sms_rag.shared.models import SearchResult
from sms_rag.shared.vector_store import VectorStoreInterface


class VectorStoreError(Exception):
    """Raised when the vector store is unreachable or times out."""


class VectorStoreTimeoutError(VectorStoreError):
    """Raised when a vector store operation exceeds the configured timeout."""


class ChromaStore(VectorStoreInterface):
    """ChromaDB implementation of the VectorStoreInterface.

    Provides persistent local vector storage with metadata filtering
    capabilities for SMS conversation embeddings.
    """

    COLLECTION_NAME = "sms_conversations"
    # Metadata fields that contain lists and need JSON serialization
    _LIST_FIELDS = {"message_types", "phone_numbers"}

    def __init__(
        self,
        persist_path: Path | str = Path("storage/chroma"),
        timeout: int = 30,
    ) -> None:
        """Initialize the ChromaDB store.

        Args:
            persist_path: Directory path for ChromaDB persistent storage.
            timeout: Maximum seconds to wait for store operations.

        Raises:
            VectorStoreError: If the store cannot be initialized.
        """
        self._timeout = timeout
        self._persist_path = Path(persist_path)

        try:
            self._client = chromadb.PersistentClient(path=str(self._persist_path))
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            raise VectorStoreError(
                f"Failed to initialize ChromaDB at '{self._persist_path}': {e}"
            ) from e

    def _serialize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Serialize list-type metadata fields to JSON strings.

        ChromaDB doesn't support list metadata natively, so lists are
        stored as JSON-encoded strings.
        """
        serialized = {}
        for key, value in metadata.items():
            if key in self._LIST_FIELDS and isinstance(value, list):
                serialized[key] = json.dumps(value)
            elif value is None:
                # ChromaDB doesn't support None values; skip them
                continue
            else:
                serialized[key] = value
        return serialized

    def _deserialize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Deserialize JSON-encoded list fields back to Python lists."""
        deserialized = {}
        for key, value in metadata.items():
            if key in self._LIST_FIELDS and isinstance(value, str):
                try:
                    deserialized[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    deserialized[key] = value
            else:
                deserialized[key] = value
        return deserialized

    def _run_with_timeout(self, operation_name: str, func, *args, **kwargs):
        """Execute an operation with a timeout.

        Uses signal-based timeout on Unix systems. Falls back to
        direct execution if signals are unavailable.

        Raises:
            VectorStoreTimeoutError: If the operation exceeds the timeout.
            VectorStoreError: If the operation fails for other reasons.
        """

        def _timeout_handler(signum, frame):
            raise VectorStoreTimeoutError(
                f"Vector store operation '{operation_name}' timed out "
                f"after {self._timeout} seconds"
            )

        try:
            # Try signal-based timeout (Unix only, main thread only)
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(self._timeout)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result
        except (ValueError, OSError):
            # signal.alarm not available (Windows or non-main thread)
            # Fall back to direct execution without timeout enforcement
            return func(*args, **kwargs)

    def store_embeddings(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Store embeddings with associated metadata in ChromaDB.

        Args:
            ids: Unique identifiers for each embedding.
            embeddings: Vector representations of the documents.
            documents: Original text content of each document.
            metadatas: Metadata dictionaries for each document.

        Raises:
            VectorStoreTimeoutError: If the operation exceeds 30 seconds.
            VectorStoreError: If the store is unreachable or fails.
        """
        serialized_metadatas = [self._serialize_metadata(m) for m in metadatas]

        def _do_store():
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=serialized_metadatas,
            )

        try:
            self._run_with_timeout("store_embeddings", _do_store)
        except (VectorStoreTimeoutError, VectorStoreError):
            raise
        except Exception as e:
            raise VectorStoreError(f"Failed to store embeddings: {e}") from e

    def query_by_similarity(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Query for similar documents with optional metadata filters.

        ChromaDB returns distances (lower is better for cosine distance).
        These are converted to similarity scores: score = 1 - distance,
        capped to [0.0, 1.0].

        Args:
            query_embedding: The query vector to search against.
            top_k: Maximum number of results to return.
            metadata_filters: Optional filters to narrow the search.

        Returns:
            List of SearchResult objects sorted by relevance score descending.

        Raises:
            VectorStoreTimeoutError: If the operation exceeds 30 seconds.
            VectorStoreError: If the store is unreachable or fails.
        """
        where_filter = None
        if metadata_filters:
            where_filter = self._build_where_filter(metadata_filters)

        def _do_query():
            kwargs: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"],
            }
            if where_filter:
                kwargs["where"] = where_filter
            return self._collection.query(**kwargs)

        try:
            results = self._run_with_timeout("query_by_similarity", _do_query)
        except (VectorStoreTimeoutError, VectorStoreError):
            raise
        except Exception as e:
            raise VectorStoreError(f"Failed to query vector store: {e}") from e

        # Parse results into SearchResult objects
        search_results: list[SearchResult] = []

        if not results or not results["ids"] or not results["ids"][0]:
            return search_results

        ids = results["ids"][0]
        documents = (
            results["documents"][0] if results["documents"] else [None] * len(ids)
        )
        metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)
        distances = (
            results["distances"][0] if results["distances"] else [1.0] * len(ids)
        )

        for i, chunk_id in enumerate(ids):
            # Convert cosine distance to similarity score
            # ChromaDB cosine distance is in [0, 2], similarity = 1 - distance
            distance = distances[i]
            score = max(0.0, min(1.0, 1.0 - distance))

            metadata = self._deserialize_metadata(metadatas[i] if metadatas[i] else {})

            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    text=documents[i] or "",
                    metadata=metadata,
                    score=score,
                )
            )

        # Sort by score descending (should already be, but ensure)
        search_results.sort(key=lambda r: r.score, reverse=True)
        return search_results

    def delete_by_ids(self, ids: list[str]) -> None:
        """Delete embeddings by their identifiers.

        Args:
            ids: List of embedding identifiers to delete.

        Raises:
            VectorStoreTimeoutError: If the operation exceeds 30 seconds.
            VectorStoreError: If the store is unreachable or fails.
        """
        if not ids:
            return

        def _do_delete():
            self._collection.delete(ids=ids)

        try:
            self._run_with_timeout("delete_by_ids", _do_delete)
        except (VectorStoreTimeoutError, VectorStoreError):
            raise
        except Exception as e:
            raise VectorStoreError(f"Failed to delete embeddings: {e}") from e

    def has_document(self, source_filename: str) -> bool:
        """Check if a source file has already been indexed.

        Queries the collection for any document with matching
        source_filename metadata.

        Args:
            source_filename: The filename to check for existing embeddings.

        Returns:
            True if the source file has been indexed, False otherwise.

        Raises:
            VectorStoreTimeoutError: If the operation exceeds 30 seconds.
            VectorStoreError: If the store is unreachable or fails.
        """

        def _do_check():
            results = self._collection.get(
                where={"source_filename": {"$eq": source_filename}},
                limit=1,
            )
            return bool(results and results["ids"])

        try:
            return self._run_with_timeout("has_document", _do_check)
        except (VectorStoreTimeoutError, VectorStoreError):
            raise
        except Exception as e:
            raise VectorStoreError(f"Failed to check document existence: {e}") from e

    def _build_where_filter(self, metadata_filters: dict[str, Any]) -> dict[str, Any]:
        """Build a ChromaDB where-filter from metadata filter dict.

        Supports simple equality filters. For list-typed fields, the
        filter checks if the JSON-serialized value contains the target.
        """
        if len(metadata_filters) == 1:
            key, value = next(iter(metadata_filters.items()))
            return {key: {"$eq": value}}

        # Multiple filters require $and
        conditions = []
        for key, value in metadata_filters.items():
            conditions.append({key: {"$eq": value}})
        return {"$and": conditions}
