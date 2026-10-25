
"""Text embeddings with Gemini and OpenAI quota fallback.

The same embedding provider must be used for document indexing and querying.
Local mode uses the Gemini API; cloud mode uses Vertex AI. If Gemini quota is
exhausted, the wrapper switches to OpenAI embeddings.
"""

from functools import lru_cache
from threading import Lock

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from config.settings import settings
from exceptions import ConfigurationError
from logger import GLOBAL_LOGGER as log



class FallbackEmbeddings(Embeddings):
    """Switch from Gemini to OpenAI when a quota-related error occurs."""

    def __init__(self, primary: Embeddings, fallback: Embeddings):
        self.primary = primary
        self.fallback = fallback
        self._use_openai = False
        self._lock = Lock()

    def _embed(self, method: str, texts):
        with self._lock:
            use_openai = self._use_openai

        if use_openai:
            return getattr(self.fallback, method)(texts)

        try:
            return getattr(self.primary, method)(texts)

        except Exception as exc:
            error = str(exc).lower()

            quota_errors = (
                "resource_exhausted",
                "prepayment credits are depleted",
                "quota exceeded",
                "429",
                "402",
            )

            if not any(term in error for term in quota_errors):
                raise

            if not settings.OPENAI_API_KEY:
                raise ConfigurationError(
                    "Gemini embedding quota exhausted. "
                    "Configure OPENAI_API_KEY in backend/.env."
                ) from exc

            with self._lock:
                self._use_openai = True

            log.warning(
                "Gemini embedding quota exhausted. Switching to OpenAI."
            )

            return getattr(self.fallback, method)(texts)

    def embed_documents(self, texts):
        return self._embed("embed_documents", texts)

    def embed_query(self, text):
        return self._embed("embed_query", text)



@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Create the configured embedding provider with quota fallback."""

    if not settings.OPENAI_API_KEY:
        raise ConfigurationError(
            "OPENAI_API_KEY is missing from backend/.env. "
            "It is required for embedding fallback."
        )

    openai_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.OPENAI_API_KEY,
        dimensions=settings.embedding_dimensions,
    )

    if settings.is_cloud:
        if not settings.GCP_PROJECT:
            raise ConfigurationError(
                "APP_MODE=cloud requires GCP_PROJECT_ID for Vertex AI embeddings."
            )

        gemini_embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model_name,
            vertexai=True,
            project=settings.GCP_PROJECT,
            location=settings.GCP_REGION,
            output_dimensionality=settings.embedding_dimensions,
        )

    elif settings.GOOGLE_API_KEY:
        gemini_embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.local_embedding_model_name,
            google_api_key=settings.GOOGLE_API_KEY,
            output_dimensionality=settings.embedding_dimensions,
        )

    else:
        log.info("GOOGLE_API_KEY is missing. Using OpenAI embeddings.")
        return openai_embeddings

    return FallbackEmbeddings(
        primary=gemini_embeddings,
        fallback=openai_embeddings,
    )
