#!/usr/bin/env python3

import json
import shutil
from typing import List, Dict, Any

import fire  # type: ignore
from tqdm import tqdm  # type: ignore
import vertexai  # type: ignore
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput  # type: ignore


def load_threads(input_file: str) -> List[Dict[str, Any]]:
    """Load threads from the processed JSON file."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    threads: List[Dict[str, Any]] = data["threads"][::-1]
    return threads


def generate_embeddings(
    input_file: str,
    output_file: str,
    batch_size: int = 32,
    project_id: str = "genai-415421",
    region: str = "us-central1",
    model_name: str = "text-multilingual-embedding-002",
) -> None:

    vertexai.init(project=project_id, location=region)

    model = TextEmbeddingModel.from_pretrained(model_name)

    print(f"Loading threads from {input_file}...")
    threads = load_threads(input_file)
    print(f"Found {len(threads)} threads")

    with open(output_file, "r", encoding="utf-8") as f:
        existing_embeddings = json.load(f)
    existing_urls = set()
    for embedding in existing_embeddings["embeddings"]:
        existing_urls.update(embedding["urls"])

    all_embeddings = existing_embeddings["embeddings"]
    new_threads = [thread for thread in threads if thread["urls"][0] not in existing_urls]
    for i in tqdm(range(0, len(new_threads), batch_size), desc="Generating embeddings"):
        batch = threads[i : i + batch_size]
        instances = [
            TextEmbeddingInput(text=thread["text"], task_type="RETRIEVAL_DOCUMENT")
            for thread in batch
        ]
        embeddings = model.get_embeddings(instances)
        for j, thread in enumerate(batch):
            embedding_data = {
                "text": thread["text"],
                "urls": thread["urls"],
                "embedding": embeddings[j].values,
            }
            all_embeddings.append(embedding_data)

        with open(output_file + "_tmp", "w", encoding="utf-8") as f:
            json.dump({"embeddings": all_embeddings}, f, ensure_ascii=False, indent=2)
        shutil.move(output_file + "_tmp", output_file)
    print("Done!")


if __name__ == "__main__":
    fire.Fire(generate_embeddings)
