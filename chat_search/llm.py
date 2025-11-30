import logging
import os
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL_NAME = "google/gemini-2.5-flash"


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
