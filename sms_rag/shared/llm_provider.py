"""Abstract interface for LLM provider operations."""

from abc import ABC, abstractmethod


class LLMProviderInterface(ABC):
    """Abstract base class for LLM provider implementations.

    Enables swapping between Ollama, Amazon Bedrock, or other LLM providers
    without modifying orchestration logic.
    """

    @abstractmethod
    def generate(self, prompt: str, context: list[str]) -> str:
        """Generate a response given a prompt and context passages.

        Args:
            prompt: The user query or instruction.
            context: List of relevant text passages to ground the response.

        Returns:
            The generated text response from the LLM.
        """
        ...
