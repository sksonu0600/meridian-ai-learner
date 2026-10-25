
"""LLM configuration for RAG and agents using OpenAI."""

from functools import lru_cache

from langchain_openai import ChatOpenAI

from config.settings import settings
from exceptions import ConfigurationError


@lru_cache(maxsize=1)
def get_llm():
    """Return the configured OpenAI chat model."""

    if not settings.OPENAI_API_KEY:
        raise ConfigurationError(
            "OPENAI_API_KEY is missing. Configure it in backend/.env."
        )

    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=settings.OPENAI_API_KEY,
        temperature=settings.llm_temperature,
    )


def extract_text(content) -> str:
    """Extract text from an LLM response."""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))

        return "\n".join(parts)

    return str(content) if content is not None else ""
