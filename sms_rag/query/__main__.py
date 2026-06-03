"""Query pipeline entry point — starts the Gradio chat interface.

Instantiates and connects all query pipeline components:
EmbeddingModel → ChromaStore → BM25Search → HybridRetriever → LLMProvider →
RAGOrchestrator → SessionStore → ChatInterface

Usage:
    python -m sms_rag.query [--host HOST] [--port PORT]
"""

import argparse
import logging
import sys
from pathlib import Path

from sms_rag.query.chat_ui import ChatInterface
from sms_rag.query.orchestrator import RAGOrchestrator
from sms_rag.query.retriever import HybridRetriever
from sms_rag.query.session_store import SessionStore
from sms_rag.shared.chroma_store import ChromaStore
from sms_rag.shared.config import ConfigValidationError, load_config
from sms_rag.shared.embedding import EmbeddingModel

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list to parse (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace with optional host and port overrides.
    """
    parser = argparse.ArgumentParser(
        description="Start the SMS RAG Gradio chat interface.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host address to bind the Gradio server to (overrides config).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port number to listen on (overrides config).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a custom config.yaml file (defaults to project root config.yaml).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Wire and launch the query pipeline.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 on success, 1 on configuration or startup errors.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args(argv)

    # Load and validate configuration
    try:
        config = load_config(config_path=args.config)
    except ConfigValidationError as exc:
        logger.error("Configuration error: %s", exc)
        return 1

    # CLI args override config values for host and port
    server_host = args.host if args.host is not None else config.server_host
    server_port = args.port if args.port is not None else config.server_port

    # Instantiate embedding model (shared between retriever components)
    logger.info("Loading embedding model: %s", config.embedding_model_name)
    embedding_model = EmbeddingModel(model_name=config.embedding_model_name)

    # Instantiate vector store
    vector_store = ChromaStore(
        persist_path=config.vector_store_path,
        timeout=config.vector_store_timeout,
    )

    # Instantiate hybrid retriever (BM25 + semantic)
    retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_model=embedding_model,
        bm25_index_path=config.bm25_index_path,
        semantic_weight=config.semantic_weight,
        top_k=config.top_k,
    )

    # Select LLM provider based on config
    if config.llm_provider == "bedrock":
        from sms_rag.query.bedrock_provider import BedrockProvider

        logger.info("Using Bedrock LLM provider (model: %s)", config.bedrock_model_id)
        llm_provider = BedrockProvider(
            model_id=config.bedrock_model_id,
            region=config.bedrock_region,
            timeout=config.llm_timeout,
        )
    else:
        from sms_rag.query.ollama_provider import OllamaProvider

        logger.info(
            "Using Ollama LLM provider (model: %s, url: %s)",
            config.ollama_model,
            config.ollama_base_url,
        )
        llm_provider = OllamaProvider(
            model=config.ollama_model,
            base_url=config.ollama_base_url,
            timeout=config.llm_timeout,
        )

    # Instantiate RAG orchestrator
    orchestrator = RAGOrchestrator(
        retriever=retriever,
        llm_provider=llm_provider,
        max_context_chunks=config.max_context_chunks,
    )

    # Instantiate session store (SQLite-backed, falls back to in-memory on failure)
    session_store = SessionStore(db_path=config.session_db_path)

    # Instantiate chat interface, wiring orchestrator and session store
    chat = ChatInterface(orchestrator=orchestrator, session_store=session_store)

    # Launch the Gradio server
    logger.info(
        "Starting SMS RAG query server on %s:%d",
        server_host,
        server_port,
    )
    chat.launch(host=server_host, port=server_port)

    return 0


if __name__ == "__main__":
    sys.exit(main())
