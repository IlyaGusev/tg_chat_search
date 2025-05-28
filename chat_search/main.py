from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging

from chat_search.embeddings import EmbeddingSearcher
from chat_search.llm import generate_text

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Chat Search API",
    description="API for semantic search over chat threads using embeddings",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embeddings searcher and LLM
embeddings_file = Path("nlp_embeddings.json")
searcher = EmbeddingSearcher(embeddings_file=embeddings_file)


class SearchQuery(BaseModel):
    query: str
    top_k: int = 10


class SearchResult(BaseModel):
    text: str
    urls: List[str]
    similarity: float


class SearchAndAnswerResponse(BaseModel):
    results: List[SearchResult]
    answer: str


PROMPT = """Based on the following chat messages, please answer the question.
Use only the information provided in the context. If you cannot answer the question
based on the given context, say so.

Answer only in Russian. Отвечай на русском языке.

Context:
{context}

Question: {query}

Answer:"""


@app.post("/search", response_model=SearchAndAnswerResponse)
async def search_and_answer(query: SearchQuery) -> SearchAndAnswerResponse:
    """Search for similar threads and generate an answer."""
    try:
        logger.info(f"Received search query: {query.query}")

        # Get relevant context through semantic search
        logger.info("Performing semantic search...")
        results: List[Dict[str, Any]] = await searcher.find_similar(query.query, query.top_k)
        logger.info(f"Found {len(results)} results")

        # Format context from search results
        context = "\n\n".join(
            [
                f"Message: {result['text']}\nSource: {', '.join(result['urls'])}"
                for result in results
            ]
        )

        logger.info("Generating answer with LLM...")
        try:
            response = await generate_text(PROMPT.format(context=context, query=query.query))
            logger.info("Successfully generated answer")
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise

        typed_results: List[SearchResult] = [SearchResult(**result) for result in results]
        return SearchAndAnswerResponse(results=typed_results, answer=response)
    except Exception as e:
        logger.error(f"Error in search_and_answer: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


# Mount static files AFTER registering API routes
app.mount("/", StaticFiles(directory="chat_search/static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)
