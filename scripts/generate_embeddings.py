#!/usr/bin/env python3

import json
import os
import shutil
from typing import Any, Dict, List

import fire  # type: ignore
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from chat_search.embedder import Embedder, gen_batch


def load_threads(input_file: str) -> List[Dict[str, Any]]:
    if input_file.endswith(".json"):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        threads: List[Dict[str, Any]] = data["threads"]
    else:
        assert input_file.endswith(".jsonl")
        with open(input_file, "r", encoding="utf-8") as f:
            threads = [json.loads(line) for line in f]
    return threads


async def generate_embeddings(
    input_file: str,
    output_embeddings_file: str,
    output_metadata_file: str,
    batch_size: int = 32,
    nrows: int = 5000,
) -> None:
    embedder = Embedder(batch_size=batch_size)

    existing_urls = set()
    all_metadata = []
    all_embeddings = None

    print("Checking existing files...")
    if os.path.exists(output_metadata_file) and os.path.exists(output_embeddings_file):
        with open(output_metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                existing_urls.update(item["urls"])
                all_metadata.append(item)
        with open(output_embeddings_file, "rb") as f:
            all_embeddings = np.load(f)["embeddings"]
        assert len(all_metadata) == len(all_embeddings)
        print(f"Found {len(all_metadata)} threads")
    else:
        print("No existing files found")

    print(f"Loading threads from {input_file}...")
    threads = list(load_threads(input_file))
    print(f"Found {len(threads)} threads")
    if nrows:
        threads = threads[:nrows]

    new_threads = [thread for thread in threads if thread["urls"][0] not in existing_urls]
    if not new_threads:
        print("No new threads found!")
        return

    print(f"Processing {len(new_threads)} new threads")

    new_embeddings_list = []
    for batch in tqdm(gen_batch(new_threads, batch_size=batch_size), desc="Generating embeddings"):
        embeddings = await embedder.embed([t["text"] for t in batch])
        for thread, embedding in zip(batch, embeddings):
            all_metadata.append(thread)
            new_embeddings_list.append(embedding)

    if new_embeddings_list:
        new_embeddings = np.array(new_embeddings_list)
        if all_embeddings is not None:
            all_embeddings = np.vstack([all_embeddings, new_embeddings])
        else:
            all_embeddings = new_embeddings

    with open(f"tmp_{output_metadata_file}", "w", encoding="utf-8") as f:
        for metadata in all_metadata:
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
    shutil.move(f"tmp_{output_metadata_file}", output_metadata_file)

    assert all_embeddings is not None, "No embeddings to save"
    print(f"Saving {len(all_embeddings)} embeddings")
    np.savez(f"tmp_{output_embeddings_file}", embeddings=all_embeddings)
    shutil.move(f"tmp_{output_embeddings_file}", output_embeddings_file)

    print(f"Saved {len(all_metadata)} items to {output_metadata_file} and {output_embeddings_file}")
    print("Done!")


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(generate_embeddings)
