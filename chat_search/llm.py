import logging
import os

from openai import AsyncOpenAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GOOGLE_API_KEY"),
)


async def generate_text(prompt: str, model_name: str = "gemini-2.5-flash-preview-04-17") -> str:
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
