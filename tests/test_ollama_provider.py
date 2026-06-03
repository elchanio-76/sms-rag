"""Unit tests for the Ollama LLM provider."""

from unittest.mock import MagicMock, patch

import pytest

from sms_rag.query.ollama_provider import OllamaProvider
from sms_rag.shared.llm_provider import LLMProviderInterface


class TestOllamaProviderInit:
    """Tests for OllamaProvider initialization."""

    def test_implements_interface(self):
        """OllamaProvider satisfies LLMProviderInterface."""
        provider = OllamaProvider()
        assert isinstance(provider, LLMProviderInterface)

    def test_default_model(self):
        """Default model is llama3.1."""
        provider = OllamaProvider()
        assert provider.model == "llama3.1"

    def test_default_base_url(self):
        """Default base URL is localhost:11434."""
        provider = OllamaProvider()
        assert provider.base_url == "http://localhost:11434"

    def test_default_timeout(self):
        """Default timeout is 60 seconds."""
        provider = OllamaProvider()
        assert provider.timeout == 60

    def test_custom_model(self):
        """Custom model name is stored."""
        provider = OllamaProvider(model="mistral")
        assert provider.model == "mistral"

    def test_custom_base_url(self):
        """Custom base URL is stored."""
        provider = OllamaProvider(base_url="http://remote:11434")
        assert provider.base_url == "http://remote:11434"

    def test_custom_timeout(self):
        """Custom timeout is stored."""
        provider = OllamaProvider(timeout=120)
        assert provider.timeout == 120


class TestOllamaProviderGenerate:
    """Tests for OllamaProvider.generate() method."""

    @patch("sms_rag.query.ollama_provider.ollama.Client")
    def test_generate_with_context(self, mock_client_class):
        """Generate sends formatted prompt with context to Ollama."""
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "The answer is 42."}}
        mock_client_class.return_value = mock_client

        provider = OllamaProvider(model="llama3.1")
        result = provider.generate(
            prompt="What is the answer?",
            context=["Passage about the answer being 42."],
        )

        assert result == "The answer is 42."
        mock_client.chat.assert_called_once()
        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["model"] == "llama3.1"
        assert len(call_kwargs["messages"]) == 1
        assert "What is the answer?" in call_kwargs["messages"][0]["content"]
        assert (
            "Passage about the answer being 42."
            in call_kwargs["messages"][0]["content"]
        )

    @patch("sms_rag.query.ollama_provider.ollama.Client")
    def test_generate_without_context(self, mock_client_class):
        """Generate with empty context sends just the prompt."""
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": "I don't have context."}
        }
        mock_client_class.return_value = mock_client

        provider = OllamaProvider()
        result = provider.generate(prompt="Hello?", context=[])

        assert result == "I don't have context."
        call_kwargs = mock_client.chat.call_args[1]
        # Without context, the prompt is sent directly
        assert call_kwargs["messages"][0]["content"] == "Hello?"

    @patch("sms_rag.query.ollama_provider.ollama.Client")
    def test_generate_multiple_context_passages(self, mock_client_class):
        """Generate formats multiple context passages with separators."""
        mock_client = MagicMock()
        mock_client.chat.return_value = {"message": {"content": "Combined answer."}}
        mock_client_class.return_value = mock_client

        provider = OllamaProvider()
        result = provider.generate(
            prompt="Tell me about both.",
            context=["First passage.", "Second passage."],
        )

        assert result == "Combined answer."
        call_kwargs = mock_client.chat.call_args[1]
        content = call_kwargs["messages"][0]["content"]
        assert "[Passage 1]" in content
        assert "[Passage 2]" in content
        assert "First passage." in content
        assert "Second passage." in content


class TestOllamaProviderErrorHandling:
    """Tests for OllamaProvider error handling."""

    @patch("sms_rag.query.ollama_provider.ollama.Client")
    def test_connection_refused_raises_runtime_error(self, mock_client_class):
        """Connection refused raises RuntimeError with descriptive message."""
        mock_client = MagicMock()
        mock_client.chat.side_effect = ConnectionError("Connection refused")
        mock_client_class.return_value = mock_client

        provider = OllamaProvider()

        with pytest.raises(RuntimeError, match="LLM service unavailable"):
            provider.generate("test", ["context"])

    @patch("sms_rag.query.ollama_provider.ollama.Client")
    def test_timeout_raises_runtime_error(self, mock_client_class):
        """Timeout raises RuntimeError with descriptive message."""
        mock_client = MagicMock()
        mock_client.chat.side_effect = Exception("Request timed out")
        mock_client_class.return_value = mock_client

        provider = OllamaProvider()

        with pytest.raises(RuntimeError, match="timed out"):
            provider.generate("test", ["context"])

    @patch("sms_rag.query.ollama_provider.ollama.Client")
    def test_response_error_raises_runtime_error(self, mock_client_class):
        """Ollama ResponseError raises RuntimeError with error details."""
        import ollama

        mock_client = MagicMock()
        mock_client.chat.side_effect = ollama.ResponseError("model not found")
        mock_client_class.return_value = mock_client

        provider = OllamaProvider()

        with pytest.raises(RuntimeError, match="LLM generation failed"):
            provider.generate("test", ["context"])

    @patch("sms_rag.query.ollama_provider.ollama.Client")
    def test_unexpected_error_raises_runtime_error(self, mock_client_class):
        """Unexpected errors raise RuntimeError without exposing internals."""
        mock_client = MagicMock()
        mock_client.chat.side_effect = ValueError("something unexpected")
        mock_client_class.return_value = mock_client

        provider = OllamaProvider()

        with pytest.raises(RuntimeError, match="unexpected error"):
            provider.generate("test", ["context"])

    @patch("sms_rag.query.ollama_provider.ollama.Client")
    def test_error_does_not_expose_internal_details(self, mock_client_class):
        """Error messages don't expose internal system details."""
        mock_client = MagicMock()
        mock_client.chat.side_effect = ConnectionError(
            "Connection refused at /var/secret/path:11434"
        )
        mock_client_class.return_value = mock_client

        provider = OllamaProvider()

        with pytest.raises(RuntimeError) as exc_info:
            provider.generate("test", ["context"])

        # The error message should not contain internal paths
        assert "/var/secret/path" not in str(exc_info.value)


class TestOllamaProviderBuildPrompt:
    """Tests for the prompt building logic."""

    def test_build_prompt_empty_context(self):
        """Empty context returns just the query."""
        provider = OllamaProvider()
        result = provider._build_prompt("What time is it?", [])
        assert result == "What time is it?"

    def test_build_prompt_single_passage(self):
        """Single context passage is properly formatted."""
        provider = OllamaProvider()
        result = provider._build_prompt("Who?", ["Alice said hello."])
        assert "Alice said hello." in result
        assert "[Passage 1]" in result
        assert "Question: Who?" in result

    def test_build_prompt_multiple_passages(self):
        """Multiple passages are numbered and separated."""
        provider = OllamaProvider()
        result = provider._build_prompt("Summary?", ["First.", "Second.", "Third."])
        assert "[Passage 1]" in result
        assert "[Passage 2]" in result
        assert "[Passage 3]" in result
        assert "---" in result
