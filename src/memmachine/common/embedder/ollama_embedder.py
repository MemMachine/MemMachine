"""
Ollama-based embedder implementation.
"""

import time
from typing import Any

import aiohttp

from memmachine.common.metrics_factory.metrics_factory import MetricsFactory

from .embedder import Embedder


class OllamaEmbedder(Embedder):
    """
    Embedder that uses Ollama's embedding models
    to generate embeddings for inputs and queries.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize an OllamaEmbedder with the provided configuration.

        Args:
            config (dict[str, Any]):
                Configuration dictionary containing:
                - model (str, optional):
                  Name of the Ollama embedding model to use
                  (default: "nomic-embed-text").
                - base_url (str, optional):
                  Base URL for the Ollama service
                  (default: "http://ollama:11434").
                - metrics_factory (MetricsFactory, optional):
                  An instance of MetricsFactory
                  for collecting usage metrics.
                - user_metrics_labels (dict[str, str], optional):
                  Labels to attach to the collected metrics.

        Raises:
            ValueError:
                If configuration argument values are missing or invalid.
            TypeError:
                If configuration argument values are of incorrect type.
        """
        super().__init__()

        self._model = config.get("model", "nomic-embed-text")
        self._base_url = config.get("base_url", "http://ollama:11434")
        self._endpoint = f"{self._base_url}/api/embeddings"

        metrics_factory = config.get("metrics_factory")
        if metrics_factory is not None and not isinstance(
            metrics_factory, MetricsFactory
        ):
            raise TypeError(
                "Metrics factory must be an instance of MetricsFactory"
            )

        self._collect_metrics = False
        if metrics_factory is not None:
            self._collect_metrics = True
            self._user_metrics_labels = config.get("user_metrics_labels", {})
            label_names = self._user_metrics_labels.keys()

            self._request_counter = metrics_factory.get_counter(
                "embedder_ollama_usage_requests",
                "Number of requests to Ollama embedder",
                label_names=label_names,
            )
            self._latency_summary = metrics_factory.get_summary(
                "embedder_ollama_latency_seconds",
                "Latency in seconds for Ollama embedder requests",
                label_names=label_names,
            )

    async def ingest_embed(self, inputs: list[Any]) -> list[list[float]]:
        return await self._embed(inputs)

    async def search_embed(self, queries: list[Any]) -> list[list[float]]:
        return await self._embed(queries)

    async def _embed(self, inputs: list[Any]) -> list[list[float]]:
        if not inputs:
            return []

        # Clean inputs (replace newlines with spaces)
        cleaned_inputs = [
            input.replace("\n", " ") if input else " " for input in inputs
        ]

        embeddings = []
        start_time = time.monotonic()

        async with aiohttp.ClientSession() as session:
            for input_text in cleaned_inputs:
                try:
                    payload = {
                        "model": self._model,
                        "prompt": input_text
                    }
                    
                    async with session.post(
                        self._endpoint,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            embeddings.append(result["embedding"])
                        else:
                            error_text = await response.text()
                            raise aiohttp.ClientError(
                                f"Ollama API error {response.status}: {error_text}"
                            )
                            
                except Exception as e:
                    raise RuntimeError(f"Failed to get embedding from Ollama: {e}")

        end_time = time.monotonic()

        if self._collect_metrics:
            self._request_counter.increment(
                value=len(inputs),
                labels=self._user_metrics_labels,
            )
            self._latency_summary.observe(
                value=end_time - start_time,
                labels=self._user_metrics_labels,
            )

        return embeddings
