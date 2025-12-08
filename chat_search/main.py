import json
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
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chat_search.db import QueryLogger
from chat_search.llm import generate_text, generate_text_stream
from chat_search.search import EmbeddingSearcher

logging.basicConfig(level=logging.INFO)
logging.getLogger("watchfiles.main").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

load_dotenv()


class Config:
    embeddings_file: Path = Path("all_embeddings.npz")
    metadata_file: Path = Path("all_meta.jsonl")
    db_file: Path = Path("queries.db")


config = Config()


def get_searcher(request: Request) -> EmbeddingSearcher:
    return cast(EmbeddingSearcher, request.app.state.searcher)


def get_query_logger(request: Request) -> QueryLogger:
    return cast(QueryLogger, request.app.state.query_logger)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("Loading embeddings and metadata...")
        app.state.searcher = EmbeddingSearcher(
            embeddings_file=config.embeddings_file, metadata_file=config.metadata_file
        )
        logger.info("Embeddings loaded")

        app.state.query_logger = QueryLogger(db_path=config.db_file)
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
    generate_summary: bool = True


class SearchResult(BaseModel):
    text: str
    urls: List[str]
    similarity: float
    source: str | None = None
    pub_time: int | None = None


class SearchAndAnswerResponse(BaseModel):
    results: List[SearchResult]
    answer: str | None = None


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


app = create_app()


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

        response = None
        if query.generate_summary:
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


@app.post("/search/stream")
async def search_and_answer_stream(
    query: SearchQuery,
    searcher: Annotated[EmbeddingSearcher, Depends(get_searcher)],
    query_logger: Annotated[QueryLogger, Depends(get_query_logger)],
) -> StreamingResponse:
    async def event_generator() -> AsyncIterator[str]:
        yield ": connected\n\n"

        error_msg = None
        results_count = None
        try:
            logger.info(f"Received streaming search query: {query.query}")

            logger.info("Performing semantic search...")
            results: List[Dict[str, Any]] = await searcher.find_similar(query.query, query.top_k)
            results_count = len(results)
            logger.info(f"Found {results_count} results")

            for result in results:
                result["pub_date"] = datetime.fromtimestamp(result["pub_time"]).strftime("%m/%d/%Y, %H:%M:%S")

            typed_results: List[SearchResult] = [SearchResult(**result) for result in results]

            yield f"event: results\ndata: {json.dumps([r.model_dump() for r in typed_results])}\n\n"

            if query.generate_summary:
                context = "\n\n".join(
                    [
                        f"===\nТекст:\n{result['text']}\nДата: {result['pub_date']}\nИсточник: {result['source']}\n==="
                        for result in results
                    ]
                )

                logger.info("Generating answer with LLM...")
                try:
                    async for chunk in generate_text_stream(PROMPT.format(context=context, query=query.query)):
                        yield f"event: answer\ndata: {json.dumps({'chunk': chunk})}\n\n"
                    logger.info("Successfully generated answer")
                except Exception as e:
                    logger.error(f"Error generating text: {str(e)}")
                    yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

            yield "event: done\ndata: {}\n\n"

            await query_logger.log_query(
                query=query.query,
                top_k=query.top_k,
                results_count=results_count,
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in search_and_answer_stream: {error_msg}")

            await query_logger.log_query(
                query=query.query,
                top_k=query.top_k,
                results_count=results_count,
                error=error_msg,
            )

            yield f"event: error\ndata: {json.dumps({'error': error_msg})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "healthy"}


app.mount("/", StaticFiles(directory="chat_search/static", html=True), name="static")


def main(
    host: str = "0.0.0.0",
    port: int = 8082,
    embeddings_file: str = "all_embeddings.npz",
    metadata_file: str = "all_meta.jsonl",
    db_file: str = "queries.db",
    reload: bool = False,
) -> None:
    config.embeddings_file = Path(embeddings_file)
    config.metadata_file = Path(metadata_file)
    config.db_file = Path(db_file)

    if reload:
        reload_includes = [
            embeddings_file,
            metadata_file,
            "chat_search/static/*.html",
            "chat_search/*.py",
        ]
        reload_excludes = [
            "*.db",
            "*.db-journal",
            "*.session",
            "*.session-journal",
            "*.pyc",
            "__pycache__",
            ".git",
            ".venv",
        ]
        uvicorn.run(
            "chat_search.main:app",
            host=host,
            port=port,
            reload=True,
            reload_includes=reload_includes,
            reload_excludes=reload_excludes,
        )
    else:
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    fire.Fire(main)
