# Implementation Plan: SMS RAG

## Overview

This plan implements a Retrieval-Augmented Generation system for SMS/WhatsApp conversation data stored as PDFs. The implementation proceeds bottom-up: shared infrastructure first, then preprocessing pipeline components, then query pipeline components, and finally integration wiring. Python with Hypothesis for property-based testing.

## Tasks

- [x] 1. Set up project structure and shared infrastructure
  - [x] 1.1 Create project skeleton with package structure and dependencies
    - Create `sms_rag/` package with `shared/`, `preprocessing/`, and `query/` subpackages
    - Create `pyproject.toml` or `requirements.txt` with all dependencies (pymupdf, sentence-transformers, chromadb, rank-bm25, langchain, gradio, ollama, boto3, hypothesis, pytest)
    - Create `__init__.py` files for all packages
    - Create `storage/` directory placeholder with `.gitkeep`
    - _Requirements: 9.1, 9.2, 9.3_

  - [x] 1.2 Implement configuration system (`sms_rag/shared/config.py`)
    - Implement `AppConfig` dataclass with all fields from the design
    - Implement YAML config loading from project root `config.yaml`
    - Implement environment variable overrides with `SMS_RAG_` prefix
    - Implement startup validation that fails with named missing/invalid parameters
    - Create a default `config.yaml` template at project root
    - _Requirements: 9.4, 9.6_

  - [x] 1.3 Write property test for configuration validation (Property 17)
    - **Property 17: Configuration Validation Errors**
    - **Validates: Requirements 9.6**

  - [x] 1.4 Implement data models and interfaces (`sms_rag/shared/`)
    - Implement `Message`, `ParsedConversation`, `ConversationChunk` dataclasses in appropriate modules
    - Implement `SearchResult`, `GenerationResult`, `PipelineSummary`, `SessionExchange` dataclasses
    - Implement `VectorStoreInterface` abstract class in `sms_rag/shared/vector_store.py`
    - Implement `LLMProviderInterface` abstract class in `sms_rag/shared/llm_provider.py`
    - _Requirements: 4.2, 6.3_

- [x] 2. Implement embedding model and vector store
  - [x] 2.1 Implement embedding model wrapper (`sms_rag/shared/embedding.py`)
    - Implement `EmbeddingModel` class with `embed_documents()` and `embed_query()` methods
    - Apply `"passage: "` prefix for documents and `"query: "` prefix for queries (e5 model requirement)
    - Normalize embeddings for cosine similarity
    - _Requirements: 2.1, 2.4_

  - [x] 2.2 Implement ChromaDB adapter (`sms_rag/shared/chroma_store.py`)
    - Implement `ChromaStore` class satisfying `VectorStoreInterface`
    - Use `chromadb.PersistentClient` with configurable path
    - Collection name: `"sms_conversations"`
    - Implement `store_embeddings()`, `query_by_similarity()`, `delete_by_ids()`, `has_document()` methods
    - Store metadata as filterable ChromaDB fields
    - Handle timeout (30s) and unreachable store with error propagation
    - _Requirements: 4.1, 4.3, 4.5_

  - [x] 2.3 Write property test for vector store metadata round-trip (Property 7)
    - **Property 7: Vector Store Metadata Round-Trip**
    - **Validates: Requirements 4.3**

- [x] 3. Implement PDF parser and conversation chunker
  - [x] 3.1 Implement PDF parser (`sms_rag/preprocessing/pdf_parser.py`)
    - Implement `PDFParser` class with `parse()` method using PyMuPDF
    - Extract text preserving chronological message order
    - Detect message boundaries via date/time stamp regex patterns
    - Extract metadata: timestamps, message types (SMS, iMessage, RCS), phone numbers
    - Derive participant name from filename (remove `.pdf` extension)
    - Handle malformed/unreadable/password-protected PDFs gracefully (log error, return empty with errors list)
    - Leave missing metadata fields as `None`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 3.2 Write property tests for PDF parsing (Properties 1, 2, 3)
    - **Property 1: Message Chronological Order Preservation**
    - **Property 2: Metadata Extraction Completeness**
    - **Property 3: Filename to Participant Name Derivation**
    - **Validates: Requirements 1.1, 1.2, 1.4**

  - [x] 3.3 Implement conversation chunker (`sms_rag/preprocessing/chunker.py`)
    - Implement `ConversationChunker` class with configurable `chunk_size` (default 20) and `overlap` (default 5)
    - Implement `chunk()` method that produces overlapping chunks respecting message boundaries
    - Generate deterministic chunk IDs: `{filename_hash}_{chunk_index}`
    - Aggregate metadata (participant name, date range, message types, phone numbers) per chunk
    - Handle conversations smaller than chunk_size (produce single chunk)
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 3.4 Write property tests for chunking (Properties 5, 6)
    - **Property 5: Chunking Structural Correctness**
    - **Property 6: Chunk Metadata Completeness**
    - **Validates: Requirements 3.1, 3.2, 3.3**

- [ ] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement BM25 index and hybrid retriever
  - [x] 5.1 Implement BM25 keyword search index (`sms_rag/preprocessing/bm25_builder.py` and `sms_rag/query/bm25_search.py`)
    - Implement BM25 index builder that tokenizes chunks via whitespace split
    - Serialize index + chunk ID mapping to pickle file at `storage/bm25_index.pkl`
    - Implement BM25 search loader that deserializes and queries the index at query time
    - _Requirements: 5.1_

  - [x] 5.2 Implement hybrid retriever (`sms_rag/query/retriever.py`)
    - Implement `HybridRetriever` class combining BM25 and semantic vector search
    - Normalize scores to [0.0, 1.0] within each method
    - Merge with configurable `semantic_weight` (default 0.5): `score = W * vector + (1-W) * bm25`
    - Deduplicate by chunk_id, keeping highest merged score
    - Return top-k results sorted by score descending
    - Apply metadata filters before search
    - Return empty result set with message when no results found
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 5.3 Write property tests for hybrid retrieval (Properties 8, 9, 10)
    - **Property 8: Hybrid Search Invokes Both Methods**
    - **Property 9: Retrieval Scoring and Ranking Invariants**
    - **Property 10: Metadata Filter Enforcement**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4**

- [x] 6. Implement LLM providers and orchestrator
  - [x] 6.1 Implement Ollama LLM provider (`sms_rag/query/ollama_provider.py`)
    - Implement `OllamaProvider` satisfying `LLMProviderInterface`
    - Connect to local Ollama via HTTP API
    - Configurable model name (default: `"llama3.1"`)
    - 60-second timeout with error propagation
    - _Requirements: 6.2, 6.3, 6.6_

  - [x] 6.2 Implement Bedrock LLM provider (`sms_rag/query/bedrock_provider.py`)
    - Implement `BedrockProvider` satisfying `LLMProviderInterface`
    - Use `boto3` Bedrock Runtime client
    - Configurable model ID and region
    - Error handling without exposing credentials
    - _Requirements: 6.3, 6.4, 6.6_

  - [x] 6.3 Implement RAG orchestrator (`sms_rag/query/orchestrator.py`)
    - Implement `RAGOrchestrator` class using LangChain for prompt templates and chain composition
    - Cap context to `max_context_chunks` (default 10) chunks passed to LLM
    - Construct prompt template with system instructions, retrieved context with source refs, and user query
    - Return `GenerationResult` with text and source chunk references (participant name, date range)
    - _Requirements: 6.1, 6.5, 6.7_

  - [x] 6.4 Write property tests for orchestrator (Properties 11, 12)
    - **Property 11: Orchestrator Context Chunk Cap**
    - **Property 12: Orchestrator Output Completeness**
    - **Validates: Requirements 6.5, 6.7**

- [x] 7. Implement session store and chat interface
  - [x] 7.1 Implement session store (`sms_rag/query/session_store.py`)
    - Implement `SessionStore` class with SQLite persistence at `storage/sessions.db`
    - Create database and schema automatically if not present
    - Implement `load_history()`, `save_exchange()`, `clear()` methods
    - FIFO eviction when exceeding 50 exchanges
    - Store `source_references` as JSON-encoded text
    - Thread-safe: use `check_same_thread=False`
    - Graceful degradation to in-memory on SQLite failures (corrupted/permission/disk full)
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

  - [x] 7.2 Write property tests for session store (Properties 18, 19, 20)
    - **Property 18: Session Persistence Round-Trip**
    - **Property 19: Session FIFO Eviction**
    - **Property 20: Session Store Graceful Degradation**
    - **Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.6**

  - [x] 7.3 Implement chat interface (`sms_rag/query/chat_ui.py`)
    - Implement `ChatInterface` class using `gr.ChatInterface`
    - Load previous session history from SessionStore on launch
    - Persist each exchange to SessionStore before displaying response
    - Display source references in collapsible section below responses
    - Reject empty/whitespace-only messages client-side
    - 60-second timeout with error display and retry option
    - Display banner if vector store is empty at startup
    - Fall back to in-memory history if SessionStore unavailable
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 9.5_

  - [x] 7.4 Write property tests for chat interface (Properties 13, 14)
    - **Property 13: Chat History Limit**
    - **Property 14: Whitespace Input Rejection**
    - **Validates: Requirements 7.3, 7.6**

- [ ] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement preprocessing pipeline entry point
  - [x] 9.1 Implement preprocessing pipeline (`sms_rag/preprocessing/pipeline.py`)
    - Implement `PreprocessingPipeline` class orchestrating: scan data dir → parse → chunk → embed → store in ChromaDB + build BM25 index
    - Skip already-indexed files (identified by source filename) unless `force_reprocess=True`
    - Continue processing remaining files on individual file failure
    - Return `PipelineSummary` with accurate counts: files_processed + files_skipped + files_errored = total
    - Print summary to stdout on completion
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 9.2 Create preprocessing entry point (`sms_rag/preprocessing/__main__.py`)
    - Implement `python -m sms_rag.preprocessing` entry point
    - Parse CLI args (data dir, force-reprocess flag)
    - Load config, instantiate all components, run pipeline
    - _Requirements: 9.2_

  - [x] 9.3 Write property tests for preprocessing pipeline (Properties 4, 15, 16)
    - **Property 4: Text Preservation Through Pipeline**
    - **Property 15: Pipeline Idempotence**
    - **Property 16: Pipeline Resilience and Summary Accuracy**
    - **Validates: Requirements 2.2, 2.4, 8.3, 8.4, 8.5**

- [x] 10. Implement query pipeline entry point and wire everything together
  - [x] 10.1 Create query pipeline entry point (`sms_rag/query/__main__.py`)
    - Implement `python -m sms_rag.query` entry point that starts Gradio server
    - Load config, instantiate retriever, orchestrator, session store, and chat UI
    - Wire all query components together
    - _Requirements: 9.3_

  - [x] 10.2 Wire preprocessing pipeline components
    - Ensure `PreprocessingPipeline` correctly instantiates and connects: PDFParser → Chunker → EmbeddingModel → ChromaStore + BM25 builder
    - Verify config flows correctly to all components
    - _Requirements: 9.1, 9.4_

  - [x] 10.3 Wire query pipeline components
    - Ensure query entry point correctly instantiates and connects: EmbeddingModel → ChromaStore → BM25Search → HybridRetriever → LLMProvider → RAGOrchestrator → SessionStore → ChatInterface
    - Verify config flows correctly to all components
    - Select LLM provider based on config (`"ollama"` or `"bedrock"`)
    - _Requirements: 9.1, 9.3, 9.4_

- [ ] 11. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties (Hypothesis, 100+ iterations each)
- Unit tests validate specific examples and edge cases
- The preprocessing and query pipelines share only `sms_rag/shared/` — no cross-imports between `preprocessing/` and `query/` at runtime
- All property tests use generators: `message_generator()`, `conversation_generator()`, `chunk_config_generator()`, `metadata_generator()`, `search_result_generator()`, `whitespace_generator()`, `config_generator()`, `session_exchange_generator()`

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.4"] },
    { "id": 2, "tasks": ["1.3", "2.1", "3.1", "3.3"] },
    { "id": 3, "tasks": ["2.2", "3.2", "3.4"] },
    { "id": 4, "tasks": ["2.3", "5.1"] },
    { "id": 5, "tasks": ["5.2", "6.1", "6.2", "7.1"] },
    { "id": 6, "tasks": ["5.3", "6.3", "7.2", "7.3"] },
    { "id": 7, "tasks": ["6.4", "7.4", "9.1"] },
    { "id": 8, "tasks": ["9.2", "9.3"] },
    { "id": 9, "tasks": ["10.1", "10.2", "10.3"] }
  ]
}
```
