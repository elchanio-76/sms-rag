"""Amazon Bedrock LLM provider implementation.

Uses boto3 Bedrock Runtime client to generate responses via the Converse API.
Configurable model ID and region with secure error handling that avoids
exposing credentials in error messages.
"""

import logging

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    NoCredentialsError,
    PartialCredentialsError,
)

from sms_rag.shared.llm_provider import LLMProviderInterface

logger = logging.getLogger(__name__)


class BedrockProvider(LLMProviderInterface):
    """LLM provider using Amazon Bedrock Runtime.

    Connects to the Bedrock Runtime service via boto3 and uses the Converse API
    to generate responses. Configurable model ID and region with a 60-second
    timeout. Authentication failures are handled without exposing credentials.
    """

    def __init__(
        self,
        model_id: str,
        region: str = "us-east-1",
        timeout: int = 60,
    ):
        """Initialize the Bedrock provider.

        Args:
            model_id: The Bedrock model identifier
                (e.g., "anthropic.claude-v2", "amazon.titan-text-express-v1").
            region: AWS region for the Bedrock service (default: "us-east-1").
            timeout: Request timeout in seconds (default: 60).
        """
        self._model_id = model_id
        self._region = region
        self._timeout = timeout

        boto_config = BotoConfig(
            region_name=region,
            read_timeout=timeout,
            connect_timeout=timeout,
            retries={"max_attempts": 1},
        )

        self._client = boto3.client(
            "bedrock-runtime",
            config=boto_config,
        )

    def generate(self, prompt: str, context: list[str]) -> str:
        """Generate a response given a prompt and context passages.

        Formats context passages into the prompt and sends to Bedrock
        via the Converse API.

        Args:
            prompt: The user query or instruction.
            context: List of relevant text passages to ground the response.

        Returns:
            The generated text response from the LLM.

        Raises:
            RuntimeError: If the Bedrock service fails to respond, with a
                descriptive message that does not expose credentials.
        """
        formatted_prompt = self._build_prompt(prompt, context)

        try:
            response = self._client.converse(
                modelId=self._model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": formatted_prompt}],
                    }
                ],
                system=[
                    {
                        "text": (
                            "You are a helpful assistant that answers questions "
                            "about SMS and WhatsApp conversations. Use only the "
                            "provided context to answer. If the context does not "
                            "contain relevant information, say so."
                        )
                    }
                ],
            )

            return self._extract_response_text(response)

        except (NoCredentialsError, PartialCredentialsError) as exc:
            logger.error("Bedrock authentication failed: credentials not configured")
            raise RuntimeError(
                "LLM service configuration error: unable to authenticate "
                "with the language model service. Please check service configuration."
            ) from exc
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            logger.error("Bedrock ClientError [%s]: %s", error_code, str(e))

            if error_code in (
                "AccessDeniedException",
                "UnrecognizedClientException",
            ):
                raise RuntimeError(
                    "LLM service configuration error: access denied. "
                    "Please check service configuration."
                ) from None

            if error_code == "ThrottlingException":
                raise RuntimeError(
                    "LLM service temporarily unavailable: rate limit exceeded. "
                    "Please try again later."
                ) from None

            if error_code == "ModelTimeoutException":
                raise RuntimeError(
                    "LLM service timeout: the model did not respond within "
                    f"{self._timeout} seconds."
                ) from None

            raise RuntimeError(
                f"LLM service error: request failed ({error_code}). "
                "Please try again later."
            ) from None

        except BotoCoreError as e:
            logger.error("Bedrock BotoCoreError: %s", str(e))
            raise RuntimeError(
                "LLM service configuration error: unable to connect to "
                "the language model service. Please check service configuration."
            ) from None

    def _build_prompt(self, prompt: str, context: list[str]) -> str:
        """Format context passages and user prompt into a single prompt string.

        Args:
            prompt: The user's question.
            context: List of relevant conversation passages.

        Returns:
            A formatted prompt string with context and question.
        """
        if not context:
            return f"Question: {prompt}"

        context_section = "\n\n---\n\n".join(
            f"Passage {i + 1}:\n{passage}" for i, passage in enumerate(context)
        )

        return (
            f"Context:\n{context_section}\n\n"
            f"Based on the context above, answer the following question:\n"
            f"Question: {prompt}"
        )

    def _extract_response_text(self, response: dict) -> str:
        """Extract the generated text from a Bedrock Converse API response.

        Args:
            response: The raw API response dictionary.

        Returns:
            The extracted text content.

        Raises:
            RuntimeError: If the response format is unexpected.
        """
        try:
            output = response["output"]["message"]["content"]
            text_parts = [block["text"] for block in output if "text" in block]
            if text_parts:
                return "".join(text_parts)

            raise RuntimeError(
                "LLM service error: received empty response from the model."
            )
        except (KeyError, TypeError, IndexError) as e:
            logger.error("Unexpected Bedrock response format: %s", str(e))
            raise RuntimeError(
                "LLM service error: unexpected response format from the model."
            ) from None
