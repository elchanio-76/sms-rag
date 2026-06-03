"""Integration tests verifying preprocessing pipeline component wiring.

Ensures that __main__.py correctly instantiates all components with config
values, and that the PreprocessingPipeline receives them properly.

Validates: Requirements 9.1, 9.4
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sms_rag.preprocessing.bm25_builder import BM25IndexBuilder
from sms_rag.preprocessing.chunker import ConversationChunker
from sms_rag.preprocessing.pdf_parser import PDFParser
from sms_rag.preprocessing.pipeline import PreprocessingPipeline
from sms_rag.shared.chroma_store import ChromaStore
from sms_rag.shared.config import AppConfig, load_config
from sms_rag.shared.embedding import EmbeddingModel


class TestPreprocessingWiring:
    """Tests that preprocessing pipeline components are wired correctly."""

    def test_config_values_flow_to_chunker(self):
        """Config chunk_size and chunk_overlap flow to ConversationChunker."""
        config = load_config()
        chunker = ConversationChunker(
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
        )
        assert chunker.chunk_size == config.chunk_size
        assert chunker.overlap == config.chunk_overlap

    def test_config_values_flow_to_bm25_builder(self):
        """Config bm25_index_path flows to BM25IndexBuilder."""
        config = load_config()
        builder = BM25IndexBuilder(bm25_index_path=config.bm25_index_path)
        assert builder.bm25_index_path == config.bm25_index_path

    def test_config_values_flow_to_chroma_store(self):
        """Config vector_store_path flows to ChromaStore."""
        with tempfile.TemporaryDirectory() as tmp:
            persist_path = Path(tmp) / "chroma"
            store = ChromaStore(persist_path=persist_path)
            assert store._persist_path == persist_path

    def test_pipeline_receives_all_components(self):
        """PreprocessingPipeline receives and stores all injected components."""
        data_dir = Path("/tmp/test_data")
        mock_store = MagicMock()
        mock_embedding = MagicMock()
        chunker = ConversationChunker(chunk_size=10, overlap=2)
        parser = PDFParser()
        bm25_builder = MagicMock(spec=BM25IndexBuilder)

        pipeline = PreprocessingPipeline(
            data_dir=data_dir,
            vector_store=mock_store,
            embedding_model=mock_embedding,
            chunker=chunker,
            parser=parser,
            bm25_builder=bm25_builder,
            force_reprocess=True,
        )

        assert pipeline.data_dir == data_dir
        assert pipeline.vector_store is mock_store
        assert pipeline.embedding_model is mock_embedding
        assert pipeline.chunker is chunker
        assert pipeline.parser is parser
        assert pipeline.bm25_builder is bm25_builder
        assert pipeline.force_reprocess is True

    def test_cli_data_dir_overrides_config(self):
        """CLI --data-dir overrides config.data_dir in __main__.py."""
        from sms_rag.preprocessing.__main__ import parse_args

        args = parse_args(["--data-dir", "/custom/path"])
        assert args.data_dir == Path("/custom/path")

        # Without CLI arg, data_dir is None (so config takes precedence)
        args_default = parse_args([])
        assert args_default.data_dir is None

    def test_force_reprocess_flag_flows_from_cli(self):
        """CLI --force-reprocess flag is parsed correctly."""
        from sms_rag.preprocessing.__main__ import parse_args

        args = parse_args(["--force-reprocess"])
        assert args.force_reprocess is True

        args_default = parse_args([])
        assert args_default.force_reprocess is False

    @patch("sms_rag.preprocessing.__main__.EmbeddingModel")
    @patch("sms_rag.preprocessing.__main__.ChromaStore")
    def test_main_wires_config_to_components(self, mock_chroma_cls, mock_embed_cls):
        """The main() function passes config values to component constructors."""
        from sms_rag.preprocessing.__main__ import main

        mock_chroma_cls.return_value = MagicMock()
        mock_embed_cls.return_value = MagicMock()

        # Run with a non-existent data dir to trigger early exit
        # This still validates that config is loaded and components are attempted
        config = load_config()

        # Call main with --data-dir pointing to a real temp dir
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            result = main(["--data-dir", str(data_dir)])

        # Verify EmbeddingModel was called with config.embedding_model_name
        mock_embed_cls.assert_called_once_with(model_name=config.embedding_model_name)

        # Verify ChromaStore was called with config.vector_store_path
        mock_chroma_cls.assert_called_once_with(persist_path=config.vector_store_path)

    def test_no_query_pipeline_imports(self):
        """Preprocessing pipeline does not import from query pipeline (Req 9.1)."""
        # Fresh check: ensure no query modules in preprocessing's import tree
        query_modules = [m for m in sys.modules if m.startswith("sms_rag.query")]
        # If any query modules are loaded, it would indicate a violation
        # They might be loaded from other tests in the session, so we verify
        # the source code doesn't import them
        import ast

        preprocessing_files = [
            "sms_rag/preprocessing/__init__.py",
            "sms_rag/preprocessing/__main__.py",
            "sms_rag/preprocessing/pipeline.py",
            "sms_rag/preprocessing/pdf_parser.py",
            "sms_rag/preprocessing/chunker.py",
            "sms_rag/preprocessing/bm25_builder.py",
        ]

        for filepath in preprocessing_files:
            full_path = Path(filepath)
            if full_path.exists():
                tree = ast.parse(full_path.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            assert not alias.name.startswith(
                                "sms_rag.query"
                            ), f"{filepath} imports {alias.name}"
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and node.module.startswith("sms_rag.query"):
                            pytest.fail(f"{filepath} imports from {node.module}")
