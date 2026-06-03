"""Embedding model wrapper for multilingual text embedding.

Uses intfloat/multilingual-e5-large by default, which supports 100+ languages
including Greek, English, and Greeklish. Embeddings are L2-normalized for
cosine similarity.
"""


class EmbeddingModel:
    """Wraps the multilingual embedding model for both indexing and query time.

    The e5 model family requires specific prefixes:
    - "passage: " for documents being indexed
    - "query: " for search queries

    The model is loaded once and reused across all operations.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents with 'passage: ' prefix for e5 models.

        Args:
            texts: List of document texts to embed.

        Returns:
            List of embedding vectors, one per input text.
            Each vector is L2-normalized for cosine similarity.
        """
        prefixed = [f"passage: {t}" for t in texts]
        return self.model.encode(prefixed, normalize_embeddings=True).tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a query with 'query: ' prefix for e5 models.

        Args:
            query: The search query text to embed.

        Returns:
            A single embedding vector, L2-normalized for cosine similarity.
        """
        return self.model.encode(f"query: {query}", normalize_embeddings=True).tolist()
