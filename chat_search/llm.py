import logging
import os
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL_NAME = "openai/gpt-5-mini"


async def generate_text(
    prompt: str,
    model_name: str = DEFAULT_MODEL_NAME,
    base_url: str = DEFAULT_BASE_URL,
    api_key: Optional[str] = None,
) -> str:
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY")

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    logger.info(f"Generating text with model: {model_name}")
    try:
        response = await client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": prompt}]
        )
        text: str = str(response.choices[0].message.content)
        return text

    except Exception as e:
        logger.error(f"Error in generate_text: {str(e)}")
        raise


async def generate_text_stream(
    prompt: str,
    model_name: str = DEFAULT_MODEL_NAME,
    base_url: str = DEFAULT_BASE_URL,
    api_key: Optional[str] = None,
) -> AsyncIterator[str]:
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY")

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    logger.info(f"Generating text stream with model: {model_name}")
    try:
        stream = await client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": prompt}], stream=True
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    except Exception as e:
        logger.error(f"Error in generate_text_stream: {str(e)}")
        raise
