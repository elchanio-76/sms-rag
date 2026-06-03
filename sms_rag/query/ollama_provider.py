"""Ollama LLM provider implementation.

Connects to a local Ollama instance via HTTP API for text generation.
Implements the LLMProviderInterface for swappable LLM provider support.
"""

from sms_rag.shared.llm_provider import LLMProviderInterface

import ollama


class OllamaProvider(LLMProviderInterface):
    """LLM provider that generates responses using a local Ollama instance.

    Args:
        model: The Ollama model name to use for generation.
        base_url: The base URL of the Ollama HTTP API.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._client = ollama.Client(host=base_url, timeout=timeout)

    def generate(self, prompt: str, context: list[str]) -> str:
        """Generate a response given a prompt and context passages.

        Formats the context passages into the prompt and sends the request
        to the local Ollama instance.

        Args:
            prompt: The user query or instruction.
            context: List of relevant text passages to ground the response.

        Returns:
            The generated text response from the LLM.

        Raises:
            RuntimeError: If Ollama is not running, times out, or returns an error.
        """
        formatted_prompt = self._build_prompt(prompt, context)

        try:
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": formatted_prompt}],
            )
            return response["message"]["content"]
        except ollama.ResponseError as e:
            raise RuntimeError(f"LLM generation failed: {e.error}") from e
        except Exception as e:
            if "timed out" in str(e).lower() or "timeout" in str(e).lower():
                raise RuntimeError(
                    "LLM request timed out after 60 seconds. "
                    "Please try again or check that Ollama is responsive."
                ) from e
            if "connection" in str(e).lower() or "refused" in str(e).lower():
                raise RuntimeError(
                    "LLM service unavailable. "
                    "Please ensure Ollama is running and accessible."
                ) from e
            raise RuntimeError(
                "LLM generation failed due to an unexpected error."
            ) from e

    def _build_prompt(self, query: str, context: list[str]) -> str:
        """Build the full prompt with context passages and the user query.

        Args:
            query: The user's question.
            context: List of relevant conversation passages.

        Returns:
            A formatted prompt string ready to send to the LLM.
        """
        if not context:
            return query

        context_block = "\n\n---\n\n".join(
            f"[Passage {i + 1}]\n{passage}" for i, passage in enumerate(context)
        )

        return (
            "Use the following conversation passages to answer the question. "
            "Base your answer only on the provided context. "
            "If the answer cannot be found in the context, say so.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {query}"
        )
