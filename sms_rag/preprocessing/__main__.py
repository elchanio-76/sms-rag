"""Entry point for the preprocessing pipeline.

Usage: python -m sms_rag.preprocessing [--data-dir PATH] [--force-reprocess]
"""

import argparse
import logging
import sys
from pathlib import Path

from sms_rag.preprocessing.bm25_builder import BM25IndexBuilder
from sms_rag.preprocessing.chunker import ConversationChunker
from sms_rag.preprocessing.pdf_parser import PDFParser
from sms_rag.preprocessing.pipeline import PreprocessingPipeline
from sms_rag.shared.chroma_store import ChromaStore
from sms_rag.shared.config import ConfigValidationError, load_config
from sms_rag.shared.embedding import EmbeddingModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list to parse (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace with data_dir and force_reprocess attributes.
    """
    parser = argparse.ArgumentParser(
        description="Run the SMS RAG preprocessing pipeline.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to the directory containing PDF files (overrides config).",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        default=False,
        help="Reprocess files even if already indexed in the vector store.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the preprocessing pipeline.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 on success, 1 on errors.
    """
    args = parse_args(argv)

    # Load configuration
    try:
        config = load_config()
    except ConfigValidationError as e:
        logger.error("Configuration error: %s", e)
        return 1

    # Determine data directory (CLI overrides config)
    data_dir = args.data_dir if args.data_dir is not None else config.data_dir

    if not data_dir.exists():
        logger.error("Data directory does not exist: %s", data_dir)
        return 1

    # Instantiate pipeline components
    parser = PDFParser()
    chunker = ConversationChunker(
        chunk_size=config.chunk_size,
        overlap=config.chunk_overlap,
    )
    embedding_model = EmbeddingModel(model_name=config.embedding_model_name)
    vector_store = ChromaStore(persist_path=config.vector_store_path)
    bm25_builder = BM25IndexBuilder(bm25_index_path=config.bm25_index_path)

    # Instantiate and run the pipeline
    pipeline = PreprocessingPipeline(
        data_dir=data_dir,
        vector_store=vector_store,
        embedding_model=embedding_model,
        chunker=chunker,
        parser=parser,
        bm25_builder=bm25_builder,
        force_reprocess=args.force_reprocess,
    )

    summary = pipeline.run()

    # Exit with code 1 if there were any errors
    if summary.files_errored > 0:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
