"""Preprocessing pipeline orchestrating PDF ingestion into the vector store.

Scans a data directory for PDF files, parses them, chunks conversations,
embeds chunks, stores them in the vector store, and builds a BM25 index.
"""

import logging
from pathlib import Path

from sms_rag.preprocessing.bm25_builder import BM25IndexBuilder
from sms_rag.preprocessing.chunker import ConversationChunker
from sms_rag.preprocessing.pdf_parser import PDFParser
from sms_rag.shared.embedding import EmbeddingModel
from sms_rag.shared.models import ConversationChunk, PipelineSummary
from sms_rag.shared.vector_store import VectorStoreInterface

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Orchestrates the full ingestion flow: scan → parse → chunk → embed → store.

    Processes all PDF files in the configured data directory, skipping
    already-indexed files unless force_reprocess is set. Individual file
    failures are logged and do not halt the pipeline.
    """

    def __init__(
        self,
        data_dir: Path,
        vector_store: VectorStoreInterface,
        embedding_model: EmbeddingModel,
        chunker: ConversationChunker,
        parser: PDFParser,
        bm25_builder: BM25IndexBuilder,
        force_reprocess: bool = False,
    ):
        """Initialize the preprocessing pipeline.

        Args:
            data_dir: Directory containing PDF files to process.
            vector_store: Vector store for embedding storage and lookup.
            embedding_model: Model for generating text embeddings.
            chunker: Chunker for splitting conversations into chunks.
            parser: PDF parser for extracting messages from PDFs.
            bm25_builder: Builder for the BM25 keyword search index.
            force_reprocess: If True, reprocess files even if already indexed.
        """
        self.data_dir = data_dir
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.chunker = chunker
        self.parser = parser
        self.bm25_builder = bm25_builder
        self.force_reprocess = force_reprocess

    def run(self) -> PipelineSummary:
        """Execute the full preprocessing pipeline.

        1. Scan data_dir for all *.pdf files
        2. For each PDF: check if indexed, parse, chunk, embed, store
        3. Build BM25 index from ALL chunks (including previously indexed)
        4. Print summary and return PipelineSummary

        Returns:
            PipelineSummary with counts of processed, skipped, and errored files.
        """
        summary = PipelineSummary()

        # Step 1: Scan data directory for PDF files
        pdf_files = sorted(self.data_dir.glob("*.pdf"))
        total_files = len(pdf_files)

        if total_files == 0:
            logger.info("No PDF files found in %s", self.data_dir)
            self._print_summary(summary, total_files)
            return summary

        # Step 2: Process each PDF file
        all_new_chunks: list[ConversationChunk] = []

        for pdf_path in pdf_files:
            try:
                # Check if already indexed
                if not self.force_reprocess and self.vector_store.has_document(
                    pdf_path.name
                ):
                    logger.info("Skipping already-indexed file: %s", pdf_path.name)
                    summary.files_skipped += 1
                    continue

                # Parse PDF
                conversation = self.parser.parse(pdf_path)

                # Check for parse errors or empty messages
                if conversation.errors or not conversation.messages:
                    error_msg = (
                        f"Failed to parse '{pdf_path.name}': "
                        f"{'; '.join(conversation.errors) if conversation.errors else 'no messages extracted'}"
                    )
                    logger.error(error_msg)
                    summary.files_errored += 1
                    summary.errors.append(error_msg)
                    continue

                # Chunk the conversation
                chunks = self.chunker.chunk(conversation)

                if not chunks:
                    error_msg = f"No chunks produced for '{pdf_path.name}'"
                    logger.error(error_msg)
                    summary.files_errored += 1
                    summary.errors.append(error_msg)
                    continue

                # Embed chunks
                texts = [chunk.text for chunk in chunks]
                embeddings = self.embedding_model.embed_documents(texts)

                # Prepare metadata for storage
                ids = [chunk.chunk_id for chunk in chunks]
                metadatas = [
                    {
                        "participant_name": chunk.participant_name,
                        "source_filename": chunk.source_filename,
                        "date_range_start": (
                            chunk.date_range_start.isoformat()
                            if chunk.date_range_start
                            else None
                        ),
                        "date_range_end": (
                            chunk.date_range_end.isoformat()
                            if chunk.date_range_end
                            else None
                        ),
                        "message_types": chunk.message_types,
                        "phone_numbers": chunk.phone_numbers,
                        "message_count": chunk.message_count,
                    }
                    for chunk in chunks
                ]

                # Store in vector store
                self.vector_store.store_embeddings(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )

                all_new_chunks.extend(chunks)
                summary.files_processed += 1
                summary.chunks_created += len(chunks)
                logger.info(
                    "Processed '%s': %d chunks created", pdf_path.name, len(chunks)
                )

            except Exception as e:
                error_msg = f"Error processing '{pdf_path.name}': {e}"
                logger.error(error_msg)
                summary.files_errored += 1
                summary.errors.append(error_msg)
                continue

        # Step 3: Build BM25 index from ALL chunks (including previously indexed)
        # We need to gather all chunks for the BM25 index, not just new ones.
        # For simplicity, we rebuild using all new chunks processed in this run
        # plus we'd need to reconstruct previously indexed chunks.
        # The design says: "After all PDFs: build BM25 index from ALL chunks
        # (including previously indexed)"
        # We rebuild the BM25 index using all chunks we can gather.
        try:
            all_chunks_for_bm25 = self._gather_all_chunks_for_bm25(all_new_chunks)
            self.bm25_builder.build(all_chunks_for_bm25)
            logger.info("BM25 index built with %d chunks", len(all_chunks_for_bm25))
        except Exception as e:
            error_msg = f"Failed to build BM25 index: {e}"
            logger.error(error_msg)
            summary.errors.append(error_msg)

        # Step 4: Print summary
        self._print_summary(summary, total_files)

        return summary

    def _gather_all_chunks_for_bm25(
        self, new_chunks: list[ConversationChunk]
    ) -> list[ConversationChunk]:
        """Gather all chunks for BM25 index building.

        Re-parses and re-chunks all PDF files to build a complete BM25 index
        covering both previously indexed and newly processed files.
        The BM25 index is rebuilt from scratch each run as per the design.

        Args:
            new_chunks: Chunks created in this pipeline run.

        Returns:
            All chunks from all PDF files in the data directory.
        """
        all_chunks: list[ConversationChunk] = []
        pdf_files = sorted(self.data_dir.glob("*.pdf"))

        # Track which filenames we already have chunks for from this run
        new_chunk_filenames = {chunk.source_filename for chunk in new_chunks}

        for pdf_path in pdf_files:
            if pdf_path.name in new_chunk_filenames:
                # Use the chunks we already created
                file_chunks = [
                    c for c in new_chunks if c.source_filename == pdf_path.name
                ]
                all_chunks.extend(file_chunks)
            else:
                # Re-parse and re-chunk previously indexed files for BM25
                try:
                    conversation = self.parser.parse(pdf_path)
                    if conversation.messages and not conversation.errors:
                        chunks = self.chunker.chunk(conversation)
                        all_chunks.extend(chunks)
                except Exception as e:
                    logger.warning(
                        "Could not re-parse '%s' for BM25 index: %s",
                        pdf_path.name,
                        e,
                    )

        return all_chunks

    def _print_summary(self, summary: PipelineSummary, total_files: int) -> None:
        """Print a pipeline run summary to stdout.

        Args:
            summary: The pipeline summary with counts.
            total_files: Total number of PDF files discovered.
        """
        print("\n" + "=" * 50)
        print("Preprocessing Pipeline Summary")
        print("=" * 50)
        print(f"Total PDF files found: {total_files}")
        print(f"Files processed:       {summary.files_processed}")
        print(f"Files skipped:         {summary.files_skipped}")
        print(f"Files errored:         {summary.files_errored}")
        print(f"Chunks created:        {summary.chunks_created}")
        if summary.errors:
            print(f"\nErrors ({len(summary.errors)}):")
            for error in summary.errors:
                print(f"  - {error}")
        print("=" * 50 + "\n")
