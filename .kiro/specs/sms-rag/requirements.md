# Requirements Document

## Introduction

This document defines the requirements for a Retrieval-Augmented Generation (RAG) application that enables conversational querying of SMS/WhatsApp conversation data. The conversations are stored as PDF files in a local directory, each named after the other participant. The system supports multilingual content (Greek, English, and Greeklish) and provides a web-based chat interface for natural language queries against the conversation database.

## Glossary

- **RAG_System**: The complete Retrieval-Augmented Generation application comprising the preprocessing pipeline, retrieval engine, LLM orchestration layer, and web chat interface
- **PDF_Parser**: The component responsible for extracting text and metadata from conversation PDF files
- **Vector_Store**: The component that stores and retrieves document embeddings (initially ChromaDB, swappable to Qdrant or Pinecone)
- **Embedding_Model**: The multilingual model that converts text chunks into vector representations (e.g., multilingual-e5-large or paraphrase-multilingual-MiniLM-L12-v2)
- **Retrieval_Engine**: The component that combines keyword search (BM25) and semantic vector search to find relevant conversation chunks
- **LLM_Orchestrator**: The framework layer (Strands or LangChain) that coordinates retrieval and language model generation
- **LLM_Provider**: The language model service used for generation (initially Ollama, swappable to Amazon Bedrock)
- **Chat_Interface**: The Gradio-based web UI that provides a conversational interface to the RAG system
- **Preprocessing_Pipeline**: The pipeline that ingests PDF files, extracts text and metadata, chunks conversations, generates embeddings, and stores them in the vector store
- **Conversation_Chunk**: A segment of conversation text suitable for embedding and retrieval, preserving message boundaries
- **Greeklish**: Greek language written phonetically using Latin alphabet characters (e.g., "Perimenoume" for "Περιμένουμε")
- **Hybrid_Search**: A retrieval strategy combining keyword-based search (BM25) with semantic vector search for improved recall
- **Session_Store**: The SQLite-based component that persists chat session history to disk, enabling conversation continuity across browser sessions and server restarts

## Requirements

### Requirement 1: PDF Parsing and Text Extraction

**User Story:** As a user, I want the system to parse my conversation PDFs automatically, so that I can query my messaging history without manual data preparation.

#### Acceptance Criteria

1. WHEN a PDF file is provided from the data directory, THE PDF_Parser SHALL extract all text content preserving the chronological order of messages, where a message boundary is identified by the presence of a date/time stamp pattern indicating the start of a new message entry
2. WHEN parsing a conversation PDF, THE PDF_Parser SHALL extract metadata including date/time stamps, message type indicators (SMS, iMessage, RCS), and phone numbers found in the text, associating each extracted metadata field with the corresponding message or conversation segment
3. IF a metadata field (date/time stamp, message type, or phone number) is not present for a given message, THEN THE PDF_Parser SHALL proceed with extraction and leave the missing metadata field empty rather than failing
4. WHEN a PDF filename is processed, THE PDF_Parser SHALL derive the conversation participant name from the filename by removing the .pdf extension and using the remaining text as the participant identifier
5. IF a PDF file is malformed, unreadable, password-protected, or contains no extractable text, THEN THE PDF_Parser SHALL log an error message identifying the filename and the reason for failure, and continue processing remaining files without interruption

### Requirement 2: Multilingual Text Handling

**User Story:** As a user, I want the system to handle Greek, English, and Greeklish text equally well, so that I can search across all my conversations regardless of language.

#### Acceptance Criteria

1. THE Embedding_Model SHALL be a multilingual model that produces vector representations where semantically equivalent content in Greek, English, and Greeklish maps to nearby regions in vector space, such that a cosine similarity of at least 0.7 is achieved between equivalent phrases across any two of the three supported languages
2. WHEN indexing conversation chunks, THE Preprocessing_Pipeline SHALL preserve the original language of the text without translation
3. WHEN a query is submitted in any of the three supported languages (Greek, English, Greeklish), THE Retrieval_Engine SHALL return at least one relevant Conversation_Chunk from the top-5 results that matches the query intent, regardless of the language used in the stored conversations
4. WHEN processing Greeklish text, THE Preprocessing_Pipeline SHALL treat Greeklish as valid input without requiring transliteration to Greek, ensuring the Embedding_Model encodes it in its original Latin-character form

### Requirement 3: Conversation Chunking

**User Story:** As a user, I want conversations to be split into meaningful segments, so that retrieved context is relevant and coherent.

#### Acceptance Criteria

1. WHEN splitting conversations into chunks, THE Preprocessing_Pipeline SHALL group consecutive messages into chunks of a configurable target size (default: 20 messages) while respecting message boundaries so that no single message is split across two chunks
2. WHEN creating chunks, THE Preprocessing_Pipeline SHALL overlap adjacent chunks by a configurable number of messages (default: 5 messages) so that conversational context is preserved across chunk boundaries
3. THE Preprocessing_Pipeline SHALL associate each Conversation_Chunk with metadata fields: participant name, date range, message type, and source filename
4. IF a conversation contains fewer messages than the configured chunk size, THEN THE Preprocessing_Pipeline SHALL produce a single chunk containing all messages in that conversation

### Requirement 4: Vector Store with Modular Design

**User Story:** As a developer, I want the vector store to be easily swappable, so that I can migrate to a different provider as needs evolve.

#### Acceptance Criteria

1. THE RAG_System SHALL use ChromaDB as the default local vector store for storing and retrieving conversation embeddings
2. THE Vector_Store SHALL implement a defined interface (abstract class or protocol) exposing at minimum: store embeddings with metadata, query by similarity with metadata filters, and delete embeddings by identifier, that allows substitution with Qdrant or Pinecone without modifying consuming code
3. WHEN storing embeddings, THE Vector_Store SHALL persist metadata (participant name, date, message type, phone numbers) as filterable fields separate from the embedding vectors
4. WHEN the vector store is changed to a different provider, THE RAG_System SHALL require only a configuration change and a new adapter implementation without modifying the retrieval or preprocessing logic
5. IF the Vector_Store is unreachable or fails to respond within 30 seconds during a store or query operation, THEN THE RAG_System SHALL return an error indication to the caller and preserve any unsaved data in memory for retry

### Requirement 5: Hybrid Search Retrieval

**User Story:** As a user, I want to find conversations using both keyword matches and semantic similarity, so that I can locate messages whether I remember exact words or just the general topic.

#### Acceptance Criteria

1. WHEN a query is submitted, THE Retrieval_Engine SHALL perform both keyword-based search (BM25) and semantic vector search against the stored conversations and return results from both methods for merging
2. WHEN combining search results, THE Retrieval_Engine SHALL merge and rank results from both keyword and semantic search using a configurable weighting parameter (0.0 = keyword only, 1.0 = semantic only) with a default value of 0.5
3. WHEN metadata filters are provided (participant name, date range), THE Retrieval_Engine SHALL apply those filters before performing the search to narrow the result set
4. THE Retrieval_Engine SHALL return the top-k most relevant Conversation_Chunks along with their metadata and relevance scores, where k is configurable with a default of 5 and a maximum of 50, and relevance scores are normalized to a 0.0–1.0 range
5. IF a query returns no matching Conversation_Chunks from either search method, THEN THE Retrieval_Engine SHALL return an empty result set with a message indicating that no relevant conversations were found

### Requirement 6: LLM Orchestration with Swappable Provider

**User Story:** As a developer, I want the LLM provider to be easily changeable, so that I can switch between local and cloud models based on cost and performance needs.

#### Acceptance Criteria

1. THE LLM_Orchestrator SHALL use a framework (Strands or LangChain) to coordinate retrieval and response generation
2. THE RAG_System SHALL use Ollama as the default LLM_Provider for local model inference
3. THE LLM_Provider SHALL implement a defined interface (abstract class or protocol) that allows substitution with Amazon Bedrock without modifying the orchestration logic
4. WHEN the LLM provider is changed, THE RAG_System SHALL require only a configuration change and a new adapter implementation without modifying the orchestration or retrieval logic
5. WHEN generating a response, THE LLM_Orchestrator SHALL pass no more than 10 retrieved Conversation_Chunks as context along with the user query to the LLM_Provider
6. IF the LLM_Provider fails to respond within 60 seconds or returns an error, THEN THE LLM_Orchestrator SHALL return an error message indicating the provider failure and the nature of the issue without exposing internal system details
7. WHEN the LLM_Provider generates a response, THE LLM_Orchestrator SHALL return the generated text answer along with references to the source Conversation_Chunks (participant name and date range) used as context

### Requirement 7: Web Chat Interface

**User Story:** As a user, I want a simple web chat interface to query my conversations, so that I can interact with the system naturally without command-line usage.

#### Acceptance Criteria

1. THE Chat_Interface SHALL provide a Gradio-based web UI with a conversational chat layout
2. WHEN a user submits a message, THE Chat_Interface SHALL send the query to the LLM_Orchestrator and display the generated response in the chat thread within 60 seconds
3. THE Chat_Interface SHALL maintain conversation history within a browser session, retaining up to 50 message exchanges (user message plus system response pairs), persisted to the Session_Store so that history is available across page refreshes and server restarts
4. THE Chat_Interface SHALL display source references (participant name, date) visually separated from the generated response text so the user can verify the information
5. IF the LLM_Orchestrator fails to return a response or the request exceeds 60 seconds, THEN THE Chat_Interface SHALL display an error message indicating the failure reason and allow the user to retry the query
6. IF a user submits an empty or whitespace-only message, THEN THE Chat_Interface SHALL not send the query to the LLM_Orchestrator and SHALL indicate that a non-empty message is required

### Requirement 8: Preprocessing Pipeline

**User Story:** As a user, I want to ingest all my conversation PDFs in one step, so that the system is ready to answer questions after initial setup.

#### Acceptance Criteria

1. WHEN the preprocessing pipeline is executed, THE Preprocessing_Pipeline SHALL scan the configured data directory for all PDF files matching the "*.pdf" extension
2. WHEN a PDF file is found that has no corresponding entries in the Vector_Store, THE Preprocessing_Pipeline SHALL parse, chunk, embed, and store the conversation data in the Vector_Store
3. IF the Vector_Store already contains data for a previously processed PDF (identified by source filename), THEN THE Preprocessing_Pipeline SHALL skip re-processing that file unless a force-reprocess flag is provided at execution time
4. IF processing of an individual PDF file fails at any stage (parsing, chunking, embedding, or storage), THEN THE Preprocessing_Pipeline SHALL log the error for that file and continue processing the remaining files
5. WHEN preprocessing completes, THE Preprocessing_Pipeline SHALL print to standard output a summary containing the number of files processed successfully, the number of files skipped, the total number of chunks created, and the number of files that encountered errors

### Requirement 9: Architecture Separation

**User Story:** As a developer, I want clear separation between the preprocessing and query pipelines, so that each can be developed, tested, and run independently.

#### Acceptance Criteria

1. THE RAG_System SHALL separate the preprocessing pipeline (ingestion, parsing, chunking, embedding, storage) from the query pipeline (retrieval, LLM generation, chat interface) such that neither pipeline imports modules or classes from the other at runtime
2. THE Preprocessing_Pipeline SHALL provide its own entry point (script or command) that can be invoked independently of the Chat_Interface and query pipeline, without requiring query pipeline components to be installed or running
3. THE Chat_Interface and query pipeline SHALL operate against an already-populated Vector_Store without requiring the Preprocessing_Pipeline to be running
4. THE RAG_System SHALL use a shared configuration system that both pipelines reference for vector store connection details, model selections, and directory paths
5. IF the query pipeline is started and the Vector_Store is empty or unreachable, THEN THE Chat_Interface SHALL display an error message indicating that no conversation data is available and that the Preprocessing_Pipeline must be run first
6. IF a required configuration value (vector store connection details, model selection, or directory path) is missing or invalid at startup, THEN THE RAG_System SHALL fail with an error message identifying the missing or invalid configuration parameter

### Requirement 10: Session History Persistence

**User Story:** As a user, I want my conversation history to be preserved across browser sessions, so that I can continue previous conversations after closing and reopening the app.

#### Acceptance Criteria

1. THE Session_Store SHALL persist chat session history (user messages and system responses) in a local SQLite database located in the storage/ directory alongside other persistent data
2. WHEN the Chat_Interface is loaded, THE Session_Store SHALL load the previous session history from the SQLite database so that prior exchanges are visible to the user
3. WHEN a new message exchange occurs, THE Session_Store SHALL persist the exchange to the SQLite database before displaying the response to the user
4. WHILE a session contains 50 message exchanges, THE Session_Store SHALL discard the oldest exchange when a new exchange is added, maintaining the 50-exchange cap
5. IF the SQLite database file does not exist at startup, THEN THE Session_Store SHALL create a new empty database with the required schema without returning an error
6. IF the Session_Store fails to read from or write to the SQLite database, THEN THE Chat_Interface SHALL continue operating with in-memory history only and log a warning indicating the persistence failure
