from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import fire  # type: ignore
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
from dotenv import load_dotenv

from chat_search.search import EmbeddingSearcher
from chat_search.llm import generate_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
app = FastAPI(
    title="Chat Search API",
    description="API for semantic search over chat threads using embeddings",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings_file = Path("all_embeddings.npz")
metadata_file = Path("all_meta.jsonl")
searcher = EmbeddingSearcher(embeddings_file=embeddings_file, metadata_file=metadata_file)


class SearchQuery(BaseModel):
    query: str
    top_k: int = 10


class SearchResult(BaseModel):
    text: str
    urls: List[str]
    similarity: float
    source: str | None = None
    pub_time: int | None = None


class SearchAndAnswerResponse(BaseModel):
    results: List[SearchResult]
    answer: str


PROMPT = """
Строго на основе представленных диалогов из чатов и постов из каналов ответь на запрос.
Используй исключительно информацию, представленную в контексте.
Если в контексте нет нужной информации для ответа на запрос, явно скажи об этом.

Отвечай исключительно на хорошем русском языке.
Отвечай коротко и сжато, но используй максимальный объём информации из контекста.
Не используй Markdown в ответе.
Не предлагай продолжения переписки - это не чат.

========
Контекст:
{context}
========

========
Запрос:
{query}
========

Твой ответ:"""


@app.post("/search", response_model=SearchAndAnswerResponse)
async def search_and_answer(query: SearchQuery) -> SearchAndAnswerResponse:
    try:
        logger.info(f"Received search query: {query.query}")

        logger.info("Performing semantic search...")
        results: List[Dict[str, Any]] = await searcher.find_similar(query.query, query.top_k)
        logger.info(f"Found {len(results)} results")

        for result in results:
            result["pub_date"] = datetime.fromtimestamp(result["pub_time"]).strftime("%m/%d/%Y, %H:%M:%S")

        context = "\n\n".join(
            [
                f"===\nТекст:\n{result['text']}\nДата: {result['pub_date']}\nИсточник: {result['source']}\n==="
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
    return {"status": "healthy"}


app.mount("/", StaticFiles(directory="chat_search/static", html=True), name="static")


def main(host: str = "0.0.0.0", port: int = 8082) -> None:
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    fire.Fire(main)
