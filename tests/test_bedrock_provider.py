"""Unit tests for the Bedrock LLM provider."""

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    NoCredentialsError,
    PartialCredentialsError,
)

from sms_rag.query.bedrock_provider import BedrockProvider
from sms_rag.shared.llm_provider import LLMProviderInterface


class TestBedrockProviderInit:
    """Tests for BedrockProvider initialization."""

    @patch("sms_rag.query.bedrock_provider.boto3.client")
    def test_implements_llm_provider_interface(self, mock_client):
        """BedrockProvider satisfies LLMProviderInterface."""
        provider = BedrockProvider(model_id="anthropic.claude-v2")
        assert isinstance(provider, LLMProviderInterface)

    @patch("sms_rag.query.bedrock_provider.boto3.client")
    def test_default_region(self, mock_client):
        """Default region is us-east-1."""
        provider = BedrockProvider(model_id="anthropic.claude-v2")
        assert provider._region == "us-east-1"

    @patch("sms_rag.query.bedrock_provider.boto3.client")
    def test_custom_region(self, mock_client):
        """Custom region is accepted."""
        provider = BedrockProvider(model_id="anthropic.claude-v2", region="eu-west-1")
        assert provider._region == "eu-west-1"

    @patch("sms_rag.query.bedrock_provider.boto3.client")
    def test_custom_timeout(self, mock_client):
        """Custom timeout is accepted."""
        provider = BedrockProvider(model_id="anthropic.claude-v2", timeout=120)
        assert provider._timeout == 120

    @patch("sms_rag.query.bedrock_provider.boto3.client")
    def test_creates_bedrock_runtime_client(self, mock_client):
        """Creates boto3 bedrock-runtime client."""
        BedrockProvider(model_id="anthropic.claude-v2")
        mock_client.assert_called_once()
        call_args = mock_client.call_args
        assert call_args[0][0] == "bedrock-runtime"


class TestBedrockProviderGenerate:
    """Tests for BedrockProvider.generate()."""

    @patch("sms_rag.query.bedrock_provider.boto3.client")
    def test_successful_generation(self, mock_boto_client):
        """Successful response returns extracted text."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "Hello, this is a response."}]}}
        }

        provider = BedrockProvider(model_id="anthropic.claude-v2")
        result = provider.generate("What did they say?", ["passage 1", "passage 2"])

        assert result == "Hello, this is a response."
        mock_client.converse.assert_called_once()

    @patch("sms_rag.query.bedrock_provider.boto3.client")
    def test_generate_with_empty_context(self, mock_boto_client):
        """Generate works with empty context list."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "No context available."}]}}
        }

        provider = BedrockProvider(model_id="anthropic.claude-v2")
        result = provider.generate("Hello?", [])

        assert result == "No context available."

    @patch("sms_rag.query.bedrock_provider.boto3.client")
    def test_multi_block_response(self, mock_boto_client):
        """Multiple text blocks are concatenated."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        mock_client.converse.return_value = {
            "output": {
                "message": {
                    "content": [
                        {"text": "Part 1. "},
                        {"text": "Part 2."},
                    ]
                }
            }
        }

        provider = BedrockProvider(model_id="anthropic.claude-v2")
        result = provider.generate("Question?", ["ctx"])

        assert result == "Part 1. Part 2."


class TestBedrockProviderErrors:
    """Tests for error handling without exposing credentials."""

    @patch("sms_rag.query.bedrock_provider.boto3.client")
    def test_no_credentials_error(self, mock_boto_client):
        """NoCredentialsError raises RuntimeError without exposing details."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        mock_client.converse.side_effect = NoCredentialsError()

        provider = BedrockProvider(model_id="anthropic.claude-v2")

        with pytest.raises(RuntimeError, match="service configuration error"):
            provider.generate("Hello?", ["ctx"])

    @patch("sms_rag.query.bedrock_provider.boto3.client")
    def test_partial_credentials_error(self, mock_boto_client):
        """PartialCredentialsError raises RuntimeError without exposing details."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        mock_client.converse.side_effect = PartialCredentialsError(
            provider="env", cred_var="AWS_SECRET_ACCESS_KEY"
        )

        provider = BedrockProvider(model_id="anthropic.claude-v2")

        with pytest.raises(RuntimeError, match="service configuration error"):
            provider.generate("Hello?", ["ctx"])

    @patch("sms_rag.query.bedrock_provider.boto3.client")
    def test_access_denied_error(self, mock_boto_client):
        """AccessDeniedException raises RuntimeError about configuration."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        mock_client.converse.side_effect = ClientError(
            error_response={"Error": {"Code": "AccessDeniedException", "Message": ""}},
            operation_name="Converse",
        )

        provider = BedrockProvider(model_id="anthropic.claude-v2")

        with pytest.raises(RuntimeError, match="service configuration error"):
            provider.generate("Hello?", ["ctx"])

    @patch("sms_rag.query.bedrock_provider.boto3.client")
    def test_throttling_error(self, mock_boto_client):
        """ThrottlingException raises RuntimeError about rate limit."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        mock_client.converse.side_effect = ClientError(
            error_response={"Error": {"Code": "ThrottlingException", "Message": ""}},
            operation_name="Converse",
        )

        provider = BedrockProvider(model_id="anthropic.claude-v2")

        with pytest.raises(RuntimeError, match="rate limit exceeded"):
            provider.generate("Hello?", ["ctx"])

    @patch("sms_rag.query.bedrock_provider.boto3.client")
    def test_timeout_error(self, mock_boto_client):
        """ModelTimeoutException raises RuntimeError about timeout."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        mock_client.converse.side_effect = ClientError(
            error_response={"Error": {"Code": "ModelTimeoutException", "Message": ""}},
            operation_name="Converse",
        )

        provider = BedrockProvider(model_id="anthropic.claude-v2")

        with pytest.raises(RuntimeError, match="timeout"):
            provider.generate("Hello?", ["ctx"])

    @patch("sms_rag.query.bedrock_provider.boto3.client")
    def test_generic_client_error(self, mock_boto_client):
        """Other ClientErrors raise RuntimeError with error code."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        mock_client.converse.side_effect = ClientError(
            error_response={
                "Error": {"Code": "ValidationException", "Message": "bad input"}
            },
            operation_name="Converse",
        )

        provider = BedrockProvider(model_id="anthropic.claude-v2")

        with pytest.raises(RuntimeError, match="ValidationException"):
            provider.generate("Hello?", ["ctx"])

    @patch("sms_rag.query.bedrock_provider.boto3.client")
    def test_botocore_error(self, mock_boto_client):
        """BotoCoreError raises RuntimeError about service configuration."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        mock_client.converse.side_effect = BotoCoreError()

        provider = BedrockProvider(model_id="anthropic.claude-v2")

        with pytest.raises(RuntimeError, match="service configuration error"):
            provider.generate("Hello?", ["ctx"])

    @patch("sms_rag.query.bedrock_provider.boto3.client")
    def test_empty_response_content(self, mock_boto_client):
        """Empty response content raises RuntimeError."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        mock_client.converse.return_value = {"output": {"message": {"content": []}}}

        provider = BedrockProvider(model_id="anthropic.claude-v2")

        with pytest.raises(RuntimeError, match="empty response"):
            provider.generate("Hello?", ["ctx"])

    @patch("sms_rag.query.bedrock_provider.boto3.client")
    def test_malformed_response(self, mock_boto_client):
        """Malformed response raises RuntimeError about unexpected format."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        mock_client.converse.return_value = {"unexpected": "format"}

        provider = BedrockProvider(model_id="anthropic.claude-v2")

        with pytest.raises(RuntimeError, match="unexpected response format"):
            provider.generate("Hello?", ["ctx"])


class TestBedrockProviderPromptBuilding:
    """Tests for prompt formatting logic."""

    @patch("sms_rag.query.bedrock_provider.boto3.client")
    def test_prompt_includes_context_passages(self, mock_boto_client):
        """Context passages are included in the formatted prompt."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "response"}]}}
        }

        provider = BedrockProvider(model_id="anthropic.claude-v2")
        provider.generate("What happened?", ["Message from Alice", "Message from Bob"])

        call_args = mock_client.converse.call_args
        messages = call_args[1]["messages"]
        user_content = messages[0]["content"][0]["text"]

        assert "Message from Alice" in user_content
        assert "Message from Bob" in user_content
        assert "What happened?" in user_content

    @patch("sms_rag.query.bedrock_provider.boto3.client")
    def test_prompt_without_context(self, mock_boto_client):
        """Without context, prompt contains just the question."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "response"}]}}
        }

        provider = BedrockProvider(model_id="anthropic.claude-v2")
        provider.generate("General question?", [])

        call_args = mock_client.converse.call_args
        messages = call_args[1]["messages"]
        user_content = messages[0]["content"][0]["text"]

        assert "General question?" in user_content
        assert "Context:" not in user_content

    @patch("sms_rag.query.bedrock_provider.boto3.client")
    def test_error_messages_do_not_contain_credentials(self, mock_boto_client):
        """Error messages never contain AWS credentials or keys."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        mock_client.converse.side_effect = NoCredentialsError()

        provider = BedrockProvider(model_id="anthropic.claude-v2")

        with pytest.raises(RuntimeError) as exc_info:
            provider.generate("Hello?", ["ctx"])

        error_msg = str(exc_info.value)
        # Should not contain typical credential-related strings
        assert "AKIA" not in error_msg
        assert "aws_secret" not in error_msg
        assert "AWS_SECRET" not in error_msg
        assert (
            "credential" not in error_msg.lower()
            or "service configuration" in error_msg.lower()
        )
