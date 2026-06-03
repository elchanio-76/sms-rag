# SMS RAG

A Retrieval-Augmented Generation (RAG) system for querying SMS/WhatsApp conversation data stored as PDFs. Ask natural language questions against your message history in Greek, English, or Greeklish via a web chat interface.

## How it works

The system runs as two independent pipelines:

1. **Preprocessing** — Scans a directory of PDF files, parses conversations, splits them into overlapping chunks, generates multilingual embeddings, and stores them in a local ChromaDB vector store alongside a BM25 keyword index.
2. **Query** — Accepts questions through a Gradio web chat UI, performs hybrid search (BM25 + semantic vector), and generates grounded answers using a local Ollama model (or Amazon Bedrock).

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally with `llama3.1` pulled (or an AWS account configured for Bedrock)

## Setup

### 1. Clone and create the virtual environment

```bash
git clone <repo-url>
cd sms-rag

python -m venv .venv
source .venv/bin/activate
```

> The project ships with a `.venv/` directory already created at `.venv`. Activate it with the command above before running any project commands.

### 2. Install dependencies

```bash
# Core dependencies
pip install -e .

# Development dependencies (tests, linting)
pip install -e ".[dev]"
```

### 3. Add your conversation PDFs

Place your exported SMS/WhatsApp PDFs in the `data/` directory. Each file should be named after the conversation participant (e.g., `John Smith.pdf`). The participant name is derived automatically from the filename.

```
data/
├── John Smith.pdf
└── Jane Doe.pdf
```

### 4. Configure

Edit `config.yaml` at the project root. The defaults work out of the box for a local Ollama setup:

```yaml
data_dir: data
llm_provider: ollama        # "ollama" or "bedrock"
ollama_model: llama3.1
ollama_base_url: http://localhost:11434
```

Any value can be overridden with an environment variable using the `SMS_RAG_` prefix:

```bash
export SMS_RAG_LLM_PROVIDER=bedrock
export SMS_RAG_BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
```

## Usage

### Step 1 — Run preprocessing

Parses, chunks, embeds, and indexes all PDFs in `data/`:

```bash
python -m sms_rag.preprocessing
```

Options:

| Flag | Description |
|------|-------------|
| `--data-dir PATH` | Override the data directory |
| `--force-reprocess` | Re-index files that were already processed |

The pipeline prints a summary when complete:

```
Files processed: 2 | Skipped: 0 | Chunks created: 84 | Errors: 0
```

Subsequent runs automatically skip already-indexed files unless `--force-reprocess` is passed.

### Step 2 — Start the chat interface

```bash
python -m sms_rag.query
```

Open `http://localhost:7860` in your browser. Chat history is persisted across restarts in `storage/sessions.db`.

Options:

| Flag | Description |
|------|-------------|
| `--host HOST` | Bind address (default: `0.0.0.0`) |
| `--port PORT` | Port (default: `7860`) |
| `--config PATH` | Path to a custom config file |

## Configuration reference

All fields in `config.yaml` with their defaults:

| Key | Default | Description |
|-----|---------|-------------|
| `data_dir` | `data` | Directory containing conversation PDFs |
| `vector_store_path` | `storage/chroma` | ChromaDB persistence directory |
| `bm25_index_path` | `storage/bm25_index.pkl` | BM25 index pickle file |
| `session_db_path` | `storage/sessions.db` | SQLite session history database |
| `embedding_model_name` | `intfloat/multilingual-e5-large` | Sentence-transformers model |
| `chunk_size` | `20` | Messages per chunk |
| `chunk_overlap` | `5` | Overlapping messages between adjacent chunks |
| `semantic_weight` | `0.5` | Balance between semantic (1.0) and keyword (0.0) search |
| `top_k` | `5` | Number of chunks to retrieve |
| `max_context_chunks` | `10` | Maximum chunks passed to the LLM |
| `llm_provider` | `ollama` | `"ollama"` or `"bedrock"` |
| `ollama_model` | `llama3.1` | Ollama model name |
| `ollama_base_url` | `http://localhost:11434` | Ollama API base URL |
| `bedrock_model_id` | _(none)_ | Required when `llm_provider` is `"bedrock"` |
| `bedrock_region` | `us-east-1` | AWS region for Bedrock |
| `server_host` | `0.0.0.0` | Gradio server bind address |
| `server_port` | `7860` | Gradio server port |
| `llm_timeout` | `60` | LLM response timeout in seconds |
| `vector_store_timeout` | `30` | Vector store operation timeout in seconds |

## Using Amazon Bedrock

Set the provider in `config.yaml` (or via environment variable) and ensure your AWS credentials are configured:

```yaml
llm_provider: bedrock
bedrock_model_id: anthropic.claude-3-sonnet-20240229-v1:0
bedrock_region: us-east-1
```

```bash
aws configure  # or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION
```

## Project structure

```
sms-rag/
├── config.yaml               # Main configuration file
├── pyproject.toml
├── data/                     # Place conversation PDFs here
├── storage/                  # Generated at runtime
│   ├── chroma/               # ChromaDB vector store
│   ├── bm25_index.pkl        # BM25 keyword index
│   └── sessions.db           # Chat session history (SQLite)
├── sms_rag/
│   ├── shared/               # Shared interfaces and utilities
│   │   ├── config.py
│   │   ├── embedding.py
│   │   ├── vector_store.py   # Abstract interface
│   │   ├── chroma_store.py   # ChromaDB implementation
│   │   └── llm_provider.py   # Abstract interface
│   ├── preprocessing/        # Ingestion pipeline
│   │   ├── __main__.py       # Entry point
│   │   ├── pipeline.py
│   │   ├── pdf_parser.py
│   │   ├── chunker.py
│   │   └── bm25_builder.py
│   └── query/                # Query pipeline
│       ├── __main__.py       # Entry point (Gradio server)
│       ├── retriever.py
│       ├── orchestrator.py
│       ├── ollama_provider.py
│       ├── bedrock_provider.py
│       ├── session_store.py
│       ├── bm25_search.py
│       └── chat_ui.py
└── tests/                    # Property-based and unit tests
```

## Running tests

```bash
# Activate the virtual environment first
source .venv/bin/activate

# Run all tests
pytest

# Run with coverage
pytest --cov=sms_rag

# Run a specific test file
pytest tests/test_chunker.py
```

Tests use [Hypothesis](https://hypothesis.readthedocs.io/) for property-based testing, running 100+ iterations per property to validate correctness invariants (chunking structure, metadata round-trips, session eviction, pipeline idempotence, etc.).

## Supported languages

The embedding model (`intfloat/multilingual-e5-large`) supports 100+ languages. The system is specifically tested for:

- **Greek** (Ελληνικά)
- **English**
- **Greeklish** — Greek written phonetically with Latin characters (e.g., "Perimenoume")

Queries and stored conversations can mix languages freely.
