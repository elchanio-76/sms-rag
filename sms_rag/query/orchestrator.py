"""RAG orchestrator coordinating retrieval and LLM generation.

Uses LangChain for prompt template composition. Retrieves context chunks
via the HybridRetriever, caps at max_context_chunks, and generates a
grounded response via the LLM provider.
"""

from langchain_core.prompts import ChatPromptTemplate

from sms_rag.query.retriever import HybridRetriever
from sms_rag.shared.llm_provider import LLMProviderInterface
from sms_rag.shared.models import GenerationResult, SearchResult


_SYSTEM_TEMPLATE = (
    "You are a helpful assistant that answers questions about SMS and WhatsApp "
    "conversations. Use ONLY the provided conversation passages to answer the "
    "user's question. If the answer cannot be found in the passages, say so clearly. "
    "When referencing information, mention the participant name and approximate date "
    "if available."
)

_CONTEXT_TEMPLATE = (
    "The following conversation passages are your source material:\n\n"
    "{context}\n\n"
    "Answer the following question based ONLY on the above passages.\n\n"
    "Question: {query}"
)


class RAGOrchestrator:
    """Coordinates retrieval and LLM generation for RAG queries.

    Retrieves candidate chunks via the HybridRetriever, caps context at
    max_context_chunks, constructs a prompt with system instructions and
    source references, then generates a response via the LLM provider.

    Args:
        retriever: HybridRetriever instance for context retrieval.
        llm_provider: LLM provider implementing LLMProviderInterface.
        max_context_chunks: Maximum number of chunks to pass as context
            to the LLM. Default: 10.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_provider: LLMProviderInterface,
        max_context_chunks: int = 10,
    ):
        self._retriever = retriever
        self._llm_provider = llm_provider
        self._max_context_chunks = max_context_chunks

        # Build LangChain prompt template
        self._prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_TEMPLATE),
                ("human", _CONTEXT_TEMPLATE),
            ]
        )

    def query(
        self, user_query: str, metadata_filters: dict | None = None
    ) -> GenerationResult:
        """Execute a RAG query: retrieve context, generate response.

        Args:
            user_query: The user's natural language question.
            metadata_filters: Optional metadata filters (participant_name,
                date range) to narrow the retrieval set.

        Returns:
            GenerationResult containing the LLM-generated text and the
            source chunk references used as context.

        Raises:
            RuntimeError: If the LLM provider fails to generate a response.
        """
        # 1. Retrieve candidate chunks
        results = self._retriever.retrieve(
            query=user_query,
            metadata_filters=metadata_filters,
            top_k=self._max_context_chunks,
        )

        # 2. Cap at max_context_chunks (take first N by score — already sorted)
        capped_results: list[SearchResult] = results[: self._max_context_chunks]

        # 3. Format context passages with source references
        context_texts = self._format_context(capped_results)

        # 4. Build the prompt using LangChain template
        prompt_messages = self._prompt_template.format_messages(
            context=context_texts,
            query=user_query,
        )
        # Extract the formatted text for the LLM provider
        formatted_context = [msg.content for msg in prompt_messages]

        # 5. Call LLM provider with the user query and context
        llm_response = self._llm_provider.generate(
            prompt=user_query,
            context=formatted_context,
        )

        # 6. Return GenerationResult with text and source chunk references
        return GenerationResult(
            text=llm_response,
            source_chunks=capped_results,
        )

    def _format_context(self, chunks: list[SearchResult]) -> str:
        """Format search result chunks into a context string with source refs.

        Each passage includes the participant name and date range extracted
        from chunk metadata for attribution.

        Args:
            chunks: List of SearchResult objects to format.

        Returns:
            Formatted context string with numbered passages and source info.
        """
        if not chunks:
            return "No relevant conversation passages found."

        passages = []
        for i, chunk in enumerate(chunks, 1):
            source_info = self._extract_source_info(chunk)
            passages.append(f"[Passage {i}] ({source_info})\n{chunk.text}")

        return "\n\n---\n\n".join(passages)

    @staticmethod
    def _extract_source_info(chunk: SearchResult) -> str:
        """Extract participant name and date range from chunk metadata.

        Args:
            chunk: A SearchResult with metadata containing source info.

        Returns:
            A formatted source reference string like
            "Participant Name, 2024-01-01 to 2024-01-15".
        """
        metadata = chunk.metadata
        participant = metadata.get("participant_name", "Unknown")
        date_start = metadata.get("date_range_start", "")
        date_end = metadata.get("date_range_end", "")

        if date_start and date_end:
            return f"{participant}, {date_start} to {date_end}"
        elif date_start:
            return f"{participant}, from {date_start}"
        elif date_end:
            return f"{participant}, until {date_end}"
        return participant
