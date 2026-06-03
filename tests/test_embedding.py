"""Tests for the EmbeddingModel class."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestEmbeddingModel:
    """Tests for EmbeddingModel prefix logic and normalization."""

    @patch("sms_rag.shared.embedding.SentenceTransformer", autospec=True)
    def _make_model(self, mock_cls):
        """Helper to create an EmbeddingModel with a mocked SentenceTransformer."""
        from sms_rag.shared.embedding import EmbeddingModel

        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        model = EmbeddingModel(model_name="test-model")
        return model, mock_instance

    def test_embed_documents_applies_passage_prefix(self):
        """Documents are prefixed with 'passage: ' before encoding."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_st = MagicMock()
            mock_cls.return_value = mock_st
            # Return a numpy array that mimics model output
            mock_st.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

            from sms_rag.shared.embedding import EmbeddingModel

            model = EmbeddingModel(model_name="test-model")
            model.embed_documents(["hello world", "test document"])

            # Verify the texts passed to encode have the passage prefix
            call_args = mock_st.encode.call_args
            assert call_args[0][0] == ["passage: hello world", "passage: test document"]
            assert call_args[1]["normalize_embeddings"] is True

    def test_embed_query_applies_query_prefix(self):
        """Queries are prefixed with 'query: ' before encoding."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_st = MagicMock()
            mock_cls.return_value = mock_st
            mock_st.encode.return_value = np.array([0.1, 0.2, 0.3])

            from sms_rag.shared.embedding import EmbeddingModel

            model = EmbeddingModel(model_name="test-model")
            model.embed_query("what did they say?")

            call_args = mock_st.encode.call_args
            assert call_args[0][0] == "query: what did they say?"
            assert call_args[1]["normalize_embeddings"] is True

    def test_embed_documents_returns_list_of_lists(self):
        """embed_documents returns a list of lists (not numpy arrays)."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_st = MagicMock()
            mock_cls.return_value = mock_st
            mock_st.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

            from sms_rag.shared.embedding import EmbeddingModel

            model = EmbeddingModel(model_name="test-model")
            result = model.embed_documents(["a", "b"])

            assert isinstance(result, list)
            assert all(isinstance(r, list) for r in result)
            assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_embed_query_returns_list(self):
        """embed_query returns a plain list (not numpy array)."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_st = MagicMock()
            mock_cls.return_value = mock_st
            mock_st.encode.return_value = np.array([0.5, 0.6, 0.7])

            from sms_rag.shared.embedding import EmbeddingModel

            model = EmbeddingModel(model_name="test-model")
            result = model.embed_query("test query")

            assert isinstance(result, list)
            assert result == [0.5, 0.6, 0.7]

    def test_embed_documents_empty_list(self):
        """embed_documents handles an empty input list."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_st = MagicMock()
            mock_cls.return_value = mock_st
            mock_st.encode.return_value = np.array([]).reshape(0, 3)

            from sms_rag.shared.embedding import EmbeddingModel

            model = EmbeddingModel(model_name="test-model")
            result = model.embed_documents([])

            call_args = mock_st.encode.call_args
            assert call_args[0][0] == []
            assert result == []

    def test_model_loaded_once_on_init(self):
        """The SentenceTransformer model is loaded once during __init__."""
        with patch("sentence_transformers.SentenceTransformer") as mock_cls:
            mock_st = MagicMock()
            mock_cls.return_value = mock_st

            from sms_rag.shared.embedding import EmbeddingModel

            model = EmbeddingModel(model_name="my-model")

            mock_cls.assert_called_once_with("my-model")
            assert model.model is mock_st
