# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a semantic search system for Telegram chat messages and channels. It downloads messages from Telegram, extracts conversation threads, generates embeddings, and provides a FastAPI web service for semantic search with LLM-powered answers in Russian.

## Development Commands

**Package management**: This project uses `uv` for dependency management.

Install dependencies:
```bash
make install
# or
uv pip install -e .
```

**Code quality**:
```bash
# Format code
make black

# Run all validation (black, flake8, mypy)
make validate

# Run tests
make test
```

**Run the web server**:
```bash
make serve
# or
python -m chat_search.main
# or with custom options
python -m chat_search.main --host=0.0.0.0 --port=8082
python -m chat_search.main --embeddings_file=my_embeddings.npz --metadata_file=my_meta.jsonl
```

CLI arguments for `main()`:
- `--host`: Server host (default: "0.0.0.0")
- `--port`: Server port (default: 8082)
- `--embeddings_file`: Path to embeddings NPZ file (default: "all_embeddings.npz")
- `--metadata_file`: Path to metadata JSONL file (default: "all_meta.jsonl")
- `--db_file`: Path to SQLite database file (default: "queries.db")

## Architecture Overview

### Data Pipeline (scripts/)

The system follows a sequential data processing pipeline:

1. **Download data** (`scripts/download_channels.py` or `scripts/download_chat.py`)
   - Downloads Telegram channel posts via web scraping (download_channels.py) or Telethon API (download_chat.py)
   - Outputs: JSONL files with individual messages

2. **Extract threads** (`scripts/extract_threads.py`)
   - Reconstructs conversation threads from flat message lists by following reply chains
   - Builds tree structures and formats threads with indentation
   - Outputs: JSONL files where each line is a complete thread with text and URLs

3. **Generate embeddings** (`scripts/generate_embeddings.py`)
   - Takes thread JSONL and generates embeddings using OpenRouter API
   - Incremental: checks existing embeddings and only processes new threads
   - Outputs:
     - `*_embeddings.npz`: NumPy array of embeddings
     - `*_meta.jsonl`: Metadata (text and URLs) for each embedding

### Search Service (chat_search/)

**Core modules**:

- `main.py`: FastAPI application with `/search` endpoint and static file serving
  - Loads embeddings and metadata at startup into `EmbeddingSearcher`
  - POST `/search`: semantic search + LLM answer generation
  - GET `/health`: health check
  - Serves static UI from `chat_search/static/`
  - Logs all queries to SQLite database (`queries.db`) via `QueryLogger`

- `search.py`: `EmbeddingSearcher` class
  - Loads embeddings (NumPy) and metadata (JSONL) at initialization
  - `find_similar()`: cosine similarity search using dot product

- `embedder.py`: `Embedder` class
  - Wraps OpenAI-compatible API (OpenRouter) for embedding generation
  - Batched processing with configurable batch size
  - Default: `google/gemini-embedding-001` model (768 dimensions)

- `llm.py`: `generate_text()` function
  - Uses OpenRouter API for text generation
  - Default: `google/gemini-2.5-flash` model
  - Configured for Russian language responses

- `db.py`: `QueryLogger` class
  - Async SQLite database wrapper using aiosqlite
  - Logs all search queries with timestamp, parameters, results count, and errors
  - Database file: `queries.db` (auto-created on startup)

### Key Data Flow

```
Telegram → download_*.py → messages.jsonl
messages.jsonl → extract_threads.py → threads.jsonl
threads.jsonl → generate_embeddings.py → embeddings.npz + meta.jsonl
embeddings.npz + meta.jsonl → (loaded by main.py) → EmbeddingSearcher
User query → /search → EmbeddingSearcher.find_similar() → LLM → Answer
```

## Environment Configuration

Required environment variables (set in `.env`):
- `OPENROUTER_API_KEY`: OpenRouter API key for embeddings and LLM
- `TG_API_ID`: Telegram API ID (for download_chat.py)
- `TG_API_HASH`: Telegram API hash (for download_chat.py)

## Important Details

**Async architecture**: All I/O operations (embedding, LLM calls, search) are async using `asyncio`. Scripts use `fire.Fire()` with async functions.

**Embedding storage**: Uses NumPy `.npz` format for efficient storage and loading of large embedding matrices. Metadata stored separately in JSONL for easy inspection and incremental updates.

**Incremental processing**: `generate_embeddings.py` tracks processed URLs and only generates embeddings for new threads, allowing efficient updates.

**Russian language**: The system is designed for Russian Telegram channels. The LLM prompt explicitly requests Russian responses.

**Data files location**: Generated data files (`.jsonl`, `.npz`) are stored in the repository root and not tracked in git.

**Type checking**: Project uses strict mypy checking (`--strict --explicit-package-bases`). Type hints are required on all functions.

**Code style**: Black formatting with 100-character line length. Flake8 ignores: E203, F403, E501, SIM115, PIE786, W503.

**Validation**: Always run `make validate` before committing changes. This runs black, flake8, and mypy with strict type checking on both scripts and chat_search directories.

**Comments**: Do not write obvious comments. Only add comments for very complicated logic that isn't self-evident from reading the code. The code should be self-documenting through clear variable and function names.
