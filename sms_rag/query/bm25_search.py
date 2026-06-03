"""BM25 keyword search module for the query pipeline.

Loads a pre-built BM25 index from a pickle file and provides a query
interface returning ranked SearchResult objects.
"""

import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from sms_rag.shared.models import SearchResult


class BM25Search:
    """Loads and queries a pre-built BM25 index."""

    def __init__(self, bm25_index_path: Path):
        """Load the BM25 index from the pickle file.

        Args:
            bm25_index_path: Path to the pickled BM25 index file.

        Raises:
            FileNotFoundError: If the index file does not exist.
        """
        with open(bm25_index_path, "rb") as f:
            data = pickle.load(f)

        self._index: BM25Okapi | None = data["index"]
        self._chunk_ids: list[str] = data["chunk_ids"]
        self._documents: list[str] = data["documents"]

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Query the BM25 index and return ranked results.

        Tokenizes the query with the same whitespace split used during
        index building, scores all documents, and returns the top-k
        results with normalized scores.

        Args:
            query: The search query string.
            top_k: Number of top results to return (default: 5).

        Returns:
            List of SearchResult objects sorted by score descending,
            with scores normalized to [0.0, 1.0].
        """
        tokenized_query = query.split()

        # Handle empty index
        if self._index is None or len(self._chunk_ids) == 0:
            return []

        scores = self._index.get_scores(tokenized_query)

        # Normalize scores to [0.0, 1.0]
        max_score = float(np.max(scores)) if len(scores) > 0 else 0.0
        if max_score > 0.0:
            normalized_scores = scores / max_score
        else:
            normalized_scores = scores

        # Get top-k indices sorted by score descending
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(normalized_scores[idx])
            # Skip zero-score results
            if score <= 0.0:
                continue
            results.append(
                SearchResult(
                    chunk_id=self._chunk_ids[idx],
                    text=self._documents[idx],
                    metadata={},
                    score=score,
                )
            )

        return results
