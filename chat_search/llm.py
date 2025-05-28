from google import genai  # type: ignore
import logging

logger = logging.getLogger(__name__)

client = genai.Client(vertexai=True, project="genai-415421", location="us-central1")


def generate_text(prompt: str, model_name: str = "gemini-2.5-flash-preview-04-17") -> str:
    logger.info(f"Generating text with model: {model_name}")
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        text: str = response.text
        return text

    except Exception as e:
        logger.error(f"Error in generate_text: {str(e)}")
        raise
