# Telegram Chat Search

Semantic search system for Telegram chat messages and channels with LLM-powered answers in Russian.

## Features

- **Semantic Search**: Find relevant messages using natural language queries, powered by embeddings
- **LLM Answers**: Get AI-generated summaries of search results in Russian
- **Thread Reconstruction**: Automatically reconstructs conversation threads from message replies
- **Incremental Processing**: Efficiently update embeddings for new messages without reprocessing everything
- **Query Analytics**: Track and analyze user search queries

## Demo

Live demo: [https://ilyagusev.dev/nlpsearch](https://ilyagusev.dev/nlpsearch)

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- OpenRouter API key
- Telegram API credentials (for downloading chat data)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd tg_chat_search
```

2. Install dependencies:
```bash
make install
# or
uv pip install -e .
```

3. Create a `.env` file with required credentials:
```bash
OPENROUTER_API_KEY=your_openrouter_api_key
TG_API_ID=your_telegram_api_id
TG_API_HASH=your_telegram_api_hash
```

## Quick Start

### Running the Web Server

If you already have embeddings and metadata files:

```bash
make serve
# or
python -m chat_search.main
```

Custom configuration:
```bash
python -m chat_search.main --host=0.0.0.0 --port=8082 \
  --embeddings_file=my_embeddings.npz \
  --metadata_file=my_meta.jsonl \
  --db_file=queries.db
```

The web interface will be available at `http://localhost:8082`

### Building Your Search Index

To create a searchable index from Telegram data, follow the data pipeline:

#### 1. Download Messages

Download from predefined Telegram channels (web scraping):
```bash
python scripts/download_channels.py <output_file.jsonl>
```

Note: Channel list is hardcoded in the script. Edit the script to customize channels.

Or download from a chat using Telethon API:
```bash
python scripts/download_chat.py <output_file.jsonl>
```

Note: Chat list is in `CHATS` dict in the script. Requires `TG_API_ID` and `TG_API_HASH` in `.env`.

#### 2. Extract Conversation Threads

Reconstruct threads from individual messages:
```bash
python scripts/extract_threads.py <input_file.jsonl> <output_file.jsonl> --min_text_length=50
```

Optional parameters:
- `--min_text_length`: Minimum thread length in characters (default: 50)

#### 3. Generate Embeddings

Create semantic embeddings for search:
```bash
python scripts/generate_embeddings.py <input_file.jsonl> <output_embeddings.npz> <output_meta.jsonl> --batch_size=32 --nrows=5000
```

Optional parameters:
- `--batch_size`: Number of threads to process at once (default: 32)
- `--nrows`: Limit number of threads to process (default: 5000, set to 0 for all)

The script is incremental - it checks existing embeddings and only processes new threads.

#### 4. Combine Multiple Sources (Optional)

If you have multiple channel/chat sources, you can combine them:
```bash
# Combine metadata files
cat source1_meta.jsonl source2_meta.jsonl > all_meta.jsonl

# Combine embeddings with Python
python -c "import numpy as np; \
  e1 = np.load('source1_embeddings.npz')['embeddings']; \
  e2 = np.load('source2_embeddings.npz')['embeddings']; \
  np.savez('all_embeddings.npz', embeddings=np.vstack([e1, e2]))"
```

## API Reference

### POST /search

Search for messages and optionally generate an AI summary.

**Request:**
```json
{
  "query": "как обучить языковую модель",
  "top_k": 15,
  "generate_summary": true
}
```

**Response:**
```json
{
  "results": [
    {
      "text": "Message thread text...",
      "urls": ["https://t.me/channel/123"],
      "similarity": 0.85,
      "source": "channel_name",
      "pub_time": 1234567890
    }
  ],
  "answer": "AI-generated summary in Russian..."
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

## Development

### Code Quality

Format code:
```bash
make black
```

Run all validation (black, flake8, mypy):
```bash
make validate
```

Run tests:
```bash
make test
```

### Query Analytics

View saved queries from the database:

```bash
# Show all queries grouped with counts
python scripts/get_queries.py all

# Show recent queries
python scripts/get_queries.py recent --limit=20

# Show database statistics
python scripts/get_queries.py stats
```

## Architecture

### Data Pipeline

```
Telegram → download_*.py → messages.jsonl
                ↓
       extract_threads.py → threads.jsonl
                ↓
    generate_embeddings.py → embeddings.npz + meta.jsonl
                ↓
            main.py (FastAPI) → EmbeddingSearcher
                ↓
         User Query → /search → Results + LLM Answer
```

### Components

**Data Processing (`scripts/`)**:
- `download_channels.py` - Web scraping for public channels
- `download_chat.py` - Telethon API for private chats/channels
- `extract_threads.py` - Thread reconstruction from reply chains
- `generate_embeddings.py` - Embedding generation via OpenRouter
- `get_queries.py` - Query analytics and reporting

**Search Service (`chat_search/`)**:
- `main.py` - FastAPI application with search endpoint
- `search.py` - `EmbeddingSearcher` class for cosine similarity search
- `embedder.py` - Embedding generation wrapper
- `llm.py` - LLM text generation for answers
- `db.py` - Query logging to SQLite

**Frontend (`chat_search/static/`)**:
- `index.html` - Single-page search interface
- `favicon.png` - Site favicon

### Tech Stack

- **Web Framework**: FastAPI + Uvicorn
- **Embeddings**: OpenRouter API (google/gemini-embedding-001)
- **LLM**: OpenRouter API (google/gemini-2.5-flash)
- **Vector Search**: NumPy cosine similarity
- **Data Format**: JSONL for metadata, NPZ for embeddings
- **Database**: SQLite for query logging
- **Telegram API**: Telethon

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENROUTER_API_KEY` | OpenRouter API key for embeddings and LLM | Yes |
| `TG_API_ID` | Telegram API ID | Yes (for download_chat.py) |
| `TG_API_HASH` | Telegram API hash | Yes (for download_chat.py) |

### CLI Arguments for main.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `0.0.0.0` | Server host |
| `--port` | `8082` | Server port |
| `--embeddings_file` | `all_embeddings.npz` | Path to embeddings file |
| `--metadata_file` | `all_meta.jsonl` | Path to metadata file |
| `--db_file` | `queries.db` | Path to SQLite database |

## License

See LICENSE file for details.

## Author

Created by [@senior_augur](https://t.me/senior_augur)

[Donate](https://t.me/senior_augur/183)
