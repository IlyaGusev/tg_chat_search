#!/usr/bin/env python3

import json
import shutil
import os
from typing import List, Dict, Any

import fire  # type: ignore
from tqdm import tqdm  # type: ignore
import vertexai  # type: ignore
import numpy as np
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput  # type: ignore


def load_threads(input_file: str) -> List[Dict[str, Any]]:
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    threads: List[Dict[str, Any]] = data["threads"][::-1]
    return threads


def generate_embeddings(
    input_file: str,
    output_embeddings_file: str,
    output_metadata_file: str,
    batch_size: int = 32,
    project_id: str = "genai-415421",
    region: str = "us-central1",
    model_name: str = "text-multilingual-embedding-002",
    nrows: int = 5000,
) -> None:

    vertexai.init(project=project_id, location=region)

    model = TextEmbeddingModel.from_pretrained(model_name)

    print(f"Loading threads from {input_file}...")
    threads = load_threads(input_file)
    print(f"Found {len(threads)} threads")

    existing_urls = set()
    all_metadata = []
    all_embeddings = np.array([])

    if os.path.exists(output_metadata_file):
        with open(output_metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                existing_urls.update(item["urls"])
                all_metadata.append(
                    {
                        "text": item["text"],
                        "urls": item["urls"],
                    }
                )

        with open(output_embeddings_file, "rb") as f:
            all_embeddings = np.load(f)["embeddings"]

    new_threads = [thread for thread in threads if thread["urls"][0] not in existing_urls][:nrows]
    print(f"Processing {len(new_threads)} new threads")

    new_embeddings_list = []
    for i in tqdm(range(0, len(new_threads), batch_size), desc="Generating embeddings"):
        batch = new_threads[i : i + batch_size]
        instances = [
            TextEmbeddingInput(text=thread["text"], task_type="RETRIEVAL_DOCUMENT")
            for thread in batch
        ]
        embeddings = model.get_embeddings(instances)

        for j, thread in enumerate(batch):
            all_metadata.append(
                {
                    "text": thread["text"],
                    "urls": thread["urls"],
                }
            )
            new_embeddings_list.append(embeddings[j].values)

    if new_embeddings_list:
        new_embeddings = np.array(new_embeddings_list)
        if all_embeddings is not None:
            all_embeddings = np.vstack([all_embeddings, new_embeddings])
        else:
            all_embeddings = new_embeddings

    with open(output_metadata_file + "_tmp", "w", encoding="utf-8") as f:
        for metadata in all_metadata:
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
    shutil.move(output_metadata_file + "_tmp", output_metadata_file)

    np.savez(output_embeddings_file + "_tmp", embeddings=all_embeddings)
    shutil.move(output_embeddings_file + "_tmp", output_embeddings_file)

    print(f"Saved {len(all_metadata)} items to {output_metadata_file} and {output_embeddings_file}")
    print("Done!")


if __name__ == "__main__":
    fire.Fire(generate_embeddings)
