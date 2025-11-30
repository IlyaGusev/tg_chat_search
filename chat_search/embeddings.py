import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput  # type: ignore
import vertexai  # type: ignore


class EmbeddingSearcher:
    def __init__(
        self,
        embeddings_file: Path,
        metadata_file: Path,
        project_id: str = "genai-415421",
        region: str = "us-central1",
        model_name: str = "text-multilingual-embedding-002",
    ):
        vertexai.init(project=project_id, location=region)
        self.model = TextEmbeddingModel.from_pretrained(model_name)

        with open(embeddings_file, "rb") as f:
            self.embeddings = np.load(f)["embeddings"]
        self.embeddings_data = []
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.embeddings_data.append(item)

    async def get_query_embedding(self, query: str) -> List[float]:
        instance = TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")
        embedding = await self.model.get_embeddings_async([instance])
        return np.array(embedding[0].values)  # type: ignore

    async def find_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = await self.get_query_embedding(query)
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

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
