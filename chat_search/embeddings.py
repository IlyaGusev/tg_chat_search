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
        project_id: str = "genai-415421",
        region: str = "us-central1",
        model_name: str = "text-multilingual-embedding-002",
    ):
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=region)
        self.model = TextEmbeddingModel.from_pretrained(model_name)

        # Load pre-computed embeddings
        with open(embeddings_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.embeddings_data = data["embeddings"]

        # Convert embeddings to numpy array for faster computation
        self.embeddings = np.array([item["embedding"] for item in self.embeddings_data])

    def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query string."""
        instance = TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")
        embedding = self.model.get_embeddings([instance])[0]
        return np.array(embedding.values)  # type: ignore

    def find_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find most similar threads to the query."""
        # Get query embedding
        query_embedding = self.get_query_embedding(query)

        # Compute cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return results
        results = []
        for idx in top_indices:
            result = {
                "text": self.embeddings_data[idx]["text"],
                "urls": self.embeddings_data[idx]["urls"],
                "similarity": float(similarities[idx]),
            }
            results.append(result)

        return results
