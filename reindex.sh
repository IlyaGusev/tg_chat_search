#!/bin/bash
set -euo pipefail

echo "Extracting chats"
uv run python -m scripts.download_chat nlp_telethon.jsonl
echo "Extracting channels"
uv run python -m scripts.download_channels channels.jsonl
cat channels.jsonl > all.jsonl
cat nlp_telethon.jsonl >> all.jsonl
echo "Extracting threads from everything"
uv run python -m scripts.extract_threads all.jsonl all_threads.jsonl
echo "Generating embeddings"
uv run python -m scripts.generate_embeddings all_threads.jsonl all_embeddings_new.npz all_meta_new.jsonl
echo "Index swap"
mv all_embeddings.npz all_embeddings_old.npz && mv all_meta.jsonl all_meta_old.jsonl
mv all_embeddings_new.npz all_embeddings.npz && mv all_meta_new.jsonl all_meta.jsonl
