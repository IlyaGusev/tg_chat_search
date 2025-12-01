import os
from typing import Any, Generator, List, Optional, cast

import numpy as np
from openai import AsyncOpenAI

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL_NAME = "google/gemini-embedding-001"
DEFAULT_BATCH_SIZE = 16
DEFAULT_EMBEDDING_DIM = 768


def gen_batch(records: List[Any], batch_size: int) -> Generator[Any, None, None]:
    batch_start = 0
    while batch_start < len(records):
        batch_end = batch_start + batch_size
        batch = records[batch_start:batch_end]
        batch_start = batch_end
        yield batch


class Embedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        api_key: Optional[str] = None,
        base_url: Optional[str] = DEFAULT_BASE_URL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    ) -> None:
        if not api_key:
            api_key = os.getenv("OPENROUTER_API_KEY")

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model_name = model_name
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim

    async def embed(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        embeddings = np.zeros((len(texts), self.embedding_dim))
        current_index = 0
        for batch in gen_batch(texts, self.batch_size):
            batch_embeddings = await self.client.embeddings.create(
                model=self.model_name,
                input=batch,
                dimensions=self.embedding_dim,
                encoding_format="float",
            )
            assert batch_embeddings.data
            for i, embedding in enumerate(batch_embeddings.data):
                assert len(embedding.embedding) == self.embedding_dim
                embeddings[current_index + i] = embedding.embedding
            current_index += self.batch_size
        return cast(List[List[float]], embeddings)
