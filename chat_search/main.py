import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, AsyncIterator, Dict, List, cast

import fire  # type: ignore
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chat_search.db import QueryLogger
from chat_search.llm import generate_text
from chat_search.search import EmbeddingSearcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


def get_searcher(request: Request) -> EmbeddingSearcher:
    return cast(EmbeddingSearcher, request.app.state.searcher)


def get_query_logger(request: Request) -> QueryLogger:
    return cast(QueryLogger, request.app.state.query_logger)


def create_app(embeddings_file: Path, metadata_file: Path, db_file: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("Loading embeddings and metadata...")
        app.state.searcher = EmbeddingSearcher(embeddings_file=embeddings_file, metadata_file=metadata_file)
        logger.info("Embeddings loaded")

        app.state.query_logger = QueryLogger(db_path=db_file)
        await app.state.query_logger.init_db()
        logger.info("Database initialized")

        yield

    app = FastAPI(
        title="Chat Search API",
        description="API for semantic search over chat threads using embeddings",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


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


def main(
    host: str = "0.0.0.0",
    port: int = 8082,
    embeddings_file: str = "all_embeddings.npz",
    metadata_file: str = "all_meta.jsonl",
    db_file: str = "queries.db",
) -> None:
    app = create_app(
        embeddings_file=Path(embeddings_file),
        metadata_file=Path(metadata_file),
        db_file=Path(db_file),
    )

    @app.post("/search", response_model=SearchAndAnswerResponse)
    async def search_and_answer(
        query: SearchQuery,
        searcher: Annotated[EmbeddingSearcher, Depends(get_searcher)],
        query_logger: Annotated[QueryLogger, Depends(get_query_logger)],
    ) -> SearchAndAnswerResponse:
        error_msg = None
        results_count = None
        try:
            logger.info(f"Received search query: {query.query}")

            logger.info("Performing semantic search...")
            results: List[Dict[str, Any]] = await searcher.find_similar(query.query, query.top_k)
            results_count = len(results)
            logger.info(f"Found {results_count} results")

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

            await query_logger.log_query(
                query=query.query,
                top_k=query.top_k,
                results_count=results_count,
            )

            return SearchAndAnswerResponse(results=typed_results, answer=response)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in search_and_answer: {error_msg}")

            await query_logger.log_query(
                query=query.query,
                top_k=query.top_k,
                results_count=results_count,
                error=error_msg,
            )

            raise HTTPException(status_code=500, detail=error_msg)

    @app.get("/health")
    async def health_check() -> Dict[str, str]:
        return {"status": "healthy"}

    app.mount("/", StaticFiles(directory="chat_search/static", html=True), name="static")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    fire.Fire(main)
