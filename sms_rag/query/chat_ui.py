"""Gradio-based web chat interface for the SMS RAG system.

Provides a conversational chat layout using gr.ChatInterface. Loads
session history on launch, persists each exchange to SessionStore,
displays source references in collapsible sections, and handles
timeouts and empty input validation.
"""

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime

import gradio as gr

from sms_rag.shared.models import GenerationResult, SessionExchange

logger = logging.getLogger(__name__)

# Timeout in seconds for orchestrator queries
_QUERY_TIMEOUT_SECONDS = 60

_EMPTY_STORE_BANNER = (
    "⚠️ **No conversation data available.** "
    "Please run the preprocessing pipeline first: "
    "`python -m sms_rag.preprocessing`"
)

_TIMEOUT_ERROR_MSG = (
    "⏱️ The request timed out after 60 seconds. "
    "Please try again with a shorter or simpler question."
)

_EMPTY_INPUT_MSG = "Please enter a non-empty message."


def _format_source_references(source_chunks: list) -> str:
    """Format source chunks as a collapsible markdown section.

    Args:
        source_chunks: List of SearchResult objects with metadata.

    Returns:
        Markdown string with a details/summary block listing sources.
    """
    if not source_chunks:
        return ""

    lines = []
    for i, chunk in enumerate(source_chunks, 1):
        metadata = chunk.metadata if hasattr(chunk, "metadata") else {}
        participant = metadata.get("participant_name", "Unknown")
        date_start = metadata.get("date_range_start", "")
        date_end = metadata.get("date_range_end", "")

        if date_start and date_end:
            date_info = f" ({date_start} – {date_end})"
        elif date_start:
            date_info = f" ({date_start})"
        else:
            date_info = ""

        lines.append(f"  {i}. **{participant}**{date_info}")

    sources_block = "\n".join(lines)
    return (
        "\n\n<details>\n<summary>📚 Sources</summary>\n\n"
        f"{sources_block}\n\n</details>"
    )


class ChatInterface:
    """Gradio-based web chat interface for the SMS RAG system.

    Uses gr.ChatInterface for conversational layout. Loads session
    history from SessionStore on launch, persists exchanges, displays
    source references, validates input, and handles timeouts.
    """

    def __init__(self, orchestrator, session_store=None):
        """Initialize the chat interface.

        Args:
            orchestrator: RAGOrchestrator instance with a query() method
                that returns GenerationResult.
            session_store: Optional SessionStore instance for persisting
                chat history. If None or unavailable, falls back to
                in-memory history.
        """
        self._orchestrator = orchestrator
        self._session_store = session_store
        self._in_memory_history: list[SessionExchange] = []
        self._session_store_available = session_store is not None
        self._vector_store_empty = False

        # Check if session store is actually functional
        if self._session_store is not None:
            try:
                self._session_store.load_history()
            except Exception as exc:
                logger.warning(
                    "SessionStore unavailable at init: %s. "
                    "Falling back to in-memory history.",
                    exc,
                )
                self._session_store_available = False

    def _check_vector_store_empty(self) -> bool:
        """Check if the vector store has any indexed data.

        Uses the orchestrator's retriever to detect an empty store.
        Returns True if the store appears empty.
        """
        try:
            # Check via public is_empty method if available and callable
            is_empty_fn = getattr(self._orchestrator, "is_store_empty", None)
            if is_empty_fn is not None and callable(is_empty_fn):
                return bool(is_empty_fn())

            # Try to access the retriever's vector store via internal attributes
            retriever = getattr(self._orchestrator, "_retriever", None)
            if retriever is not None:
                store = getattr(retriever, "_vector_store", None)
                if store is not None:
                    collection = getattr(store, "_collection", None)
                    if collection is not None:
                        count = collection.count()
                        return count == 0
            # Fallback: assume store is not empty
            return False
        except Exception:  # noqa: BLE001
            logger.warning("Could not check vector store status", exc_info=True)
            return True

    def _load_session_history(self) -> list[dict]:
        """Load session history as Gradio chat messages.

        Returns:
            List of message dicts in Gradio's messages format:
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        """
        exchanges: list[SessionExchange] = []

        if self._session_store_available:
            try:
                exchanges = self._session_store.load_history()
            except Exception as exc:
                logger.warning(
                    "Failed to load session history: %s. Using in-memory.",
                    exc,
                )
                self._session_store_available = False
                exchanges = list(self._in_memory_history)
        else:
            exchanges = list(self._in_memory_history)

        messages = []
        for exchange in exchanges:
            messages.append({"role": "user", "content": exchange.user_message})
            # Reconstruct response with sources if available
            response_text = exchange.assistant_response
            messages.append({"role": "assistant", "content": response_text})

        return messages

    def _persist_exchange(
        self, user_message: str, assistant_response: str, source_chunks: list
    ) -> None:
        """Persist a new exchange to the session store.

        Falls back to in-memory storage if the session store is unavailable.

        Args:
            user_message: The user's original message.
            assistant_response: The full assistant response (including sources).
            source_chunks: List of SearchResult objects used as sources.
        """
        source_refs = []
        for chunk in source_chunks:
            metadata = chunk.metadata if hasattr(chunk, "metadata") else {}
            source_refs.append(
                {
                    "chunk_id": chunk.chunk_id if hasattr(chunk, "chunk_id") else "",
                    "participant_name": metadata.get("participant_name", ""),
                    "date_range_start": metadata.get("date_range_start", ""),
                    "date_range_end": metadata.get("date_range_end", ""),
                }
            )

        exchange = SessionExchange(
            exchange_id=str(uuid.uuid4()),
            user_message=user_message,
            assistant_response=assistant_response,
            source_references=source_refs,
            timestamp=datetime.now(),
        )

        if self._session_store_available:
            try:
                self._session_store.save_exchange(exchange)
            except Exception as exc:
                logger.warning(
                    "Failed to persist exchange: %s. Falling back to in-memory.",
                    exc,
                )
                self._session_store_available = False
                self._in_memory_history.append(exchange)
                while len(self._in_memory_history) > 50:
                    self._in_memory_history.pop(0)
        else:
            self._in_memory_history.append(exchange)
            while len(self._in_memory_history) > 50:
                self._in_memory_history.pop(0)

    def _respond(self, message: str, history: list[dict]) -> str:
        """Process a user message and return an assistant response.

        This is the main callback for gr.ChatInterface. Validates input,
        calls the orchestrator with a timeout, formats the response with
        source references, and persists the exchange.

        Args:
            message: The user's input message.
            history: The current chat history (Gradio messages format).

        Returns:
            The assistant's response string (with source references).
        """
        # Reject empty/whitespace-only messages
        if not message or not message.strip():
            return _EMPTY_INPUT_MSG

        # Execute query with timeout
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._orchestrator.query, message.strip())
                result: GenerationResult = future.result(timeout=_QUERY_TIMEOUT_SECONDS)
        except FuturesTimeoutError:
            return _TIMEOUT_ERROR_MSG
        except Exception as exc:
            logger.error("Orchestrator query failed: %s", exc)
            return f"❌ An error occurred: {exc}"

        # Format response with source references
        response_text = result.text
        sources_section = _format_source_references(result.source_chunks)
        full_response = response_text + sources_section

        # Persist exchange before displaying
        self._persist_exchange(message.strip(), full_response, result.source_chunks)

        return full_response

    def launch(self, host: str = "0.0.0.0", port: int = 7860) -> None:
        """Launch the Gradio chat interface.

        Args:
            host: The host address to bind to.
            port: The port number to listen on.
        """
        # Check vector store status
        self._vector_store_empty = self._check_vector_store_empty()

        # Load session history for initial display
        initial_history = self._load_session_history()

        # Build the Gradio interface
        with gr.Blocks(title="SMS RAG Chat") as demo:
            if self._vector_store_empty:
                gr.Markdown(_EMPTY_STORE_BANNER)

            chatbot = gr.Chatbot(
                value=initial_history,
                label="SMS RAG",
                show_label=True,
            )

            gr.ChatInterface(
                fn=self._respond,
                chatbot=chatbot,
                title="SMS RAG Chat",
                description=(
                    "Ask questions about your SMS/WhatsApp conversations. "
                    "Supports Greek, English, and Greeklish."
                ),
            )

        demo.launch(server_name=host, server_port=port)
