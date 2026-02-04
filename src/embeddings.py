"""Embeddings module using Jina AI."""

import os
from langchain_core.embeddings import Embeddings
import requests


class JinaEmbeddings(Embeddings):
    """
    Jina AI embeddings implementation for LangChain.

    Uses Jina AI's embedding API to generate vector representations
    of text for semantic search.
    """

    def __init__(self, api_key: str | None = None, model: str = "jina-embeddings-v3"):
        """
        Initialize Jina embeddings.

        Args:
            api_key: Jina AI API key. If not provided, reads from JINA_API_KEY env var.
            model: Model name to use for embeddings.
        """
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Jina API key is required. Set JINA_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.model = model
        self.api_url = "https://api.jina.ai/v1/embeddings"

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Get embeddings for a list of texts from Jina API.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model,
            "input": texts,
        }

        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()

        result = response.json()
        embeddings = [item["embedding"] for item in result["data"]]
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of document texts to embed.

        Returns:
            List of embedding vectors.
        """
        return self._get_embeddings(texts)

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query text.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector for the query.
        """
        return self._get_embeddings([text])[0]


def get_embeddings(api_key: str | None = None) -> Embeddings:
    """
    Create and return a Jina AI embeddings instance.

    Args:
        api_key: Optional Jina AI API key.

    Returns:
        Configured embeddings instance.
    """
    return JinaEmbeddings(api_key=api_key)
