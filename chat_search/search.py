import json
import datetime
from pathlib import Path
from typing import List, Any, Dict

import numpy as np

from chat_search.embedder import Embedder


class EmbeddingSearcher:
    def __init__(
        self,
        embeddings_file: Path,
        metadata_file: Path,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.embedder = Embedder(*args, **kwargs)

        with open(embeddings_file, "rb") as f:
            self.embeddings = np.load(f)["embeddings"]

        self.embeddings_data = []
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.embeddings_data.append(item)

    async def get_query_embedding(self, query: str) -> List[float]:
        embeddings = await self.embedder.embed([query])
        return embeddings[0]

    async def find_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = await self.get_query_embedding(query)
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        max_timestamp = max([r["pub_time"] for r in self.embeddings_data])
        timestamp_diffs_days = [(max_timestamp - r["pub_time"]) // 86400 for r in self.embeddings_data]
        time_penalties = np.array([0.8 + 0.2 * (max(365 - d, 0) / 365) for d in timestamp_diffs_days])
        similarities = np.multiply(similarities, time_penalties)

        length_penalties = np.array([0.7 + 0.3 * min(len(thread["text"]), 300) / 300 for thread in self.embeddings_data])
        similarities = np.multiply(similarities, length_penalties)

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            result = {
                "text": self.embeddings_data[idx]["text"],
                "urls": self.embeddings_data[idx]["urls"],
                "similarity": float(similarities[idx]),
            }
            results.append(result)

        return results
