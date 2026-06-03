"""BM25 index builder for keyword search during preprocessing.

Tokenizes conversation chunks via whitespace split and serializes the BM25 index
along with chunk ID mapping to a pickle file for use at query time.
"""

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from sms_rag.shared.models import ConversationChunk


class BM25IndexBuilder:
    """Builds a BM25 index from conversation chunks and persists it to disk."""

    def __init__(self, bm25_index_path: Path):
        """Initialize the builder with the target output path.

        Args:
            bm25_index_path: Path where the pickled BM25 index will be saved.
        """
        self.bm25_index_path = bm25_index_path

    def build(self, chunks: list[ConversationChunk]) -> None:
        """Build the BM25 index from chunks and serialize to disk.

        Tokenizes each chunk's text via whitespace split, builds the BM25Okapi
        index, and saves the index along with chunk IDs and document texts
        as a pickle file.

        Args:
            chunks: List of ConversationChunk objects to index.
        """
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        tokenized_corpus = [doc.split() for doc in documents]

        # BM25Okapi raises ZeroDivisionError on empty corpus
        index = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

        data = {
            "index": index,
            "chunk_ids": chunk_ids,
            "documents": documents,
        }

        # Ensure the parent directory exists
        self.bm25_index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.bm25_index_path, "wb") as f:
            pickle.dump(data, f)
