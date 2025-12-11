"""Helpers for building long-term memory from configuration."""

from pydantic import InstanceOf

from memmachine.common.configuration.episodic_config import LongTermMemoryConf
from memmachine.common.resource_manager import CommonResourceManager

from .long_term_memory import LongTermMemoryParams


async def long_term_memory_params_from_config(
    config: LongTermMemoryConf,
    resource_manager: InstanceOf[CommonResourceManager],
) -> LongTermMemoryParams:
    """Build LongTermMemory parameters from configuration and resources."""
    vector_graph_store = await resource_manager.get_vector_graph_store(
        config.vector_graph_store,
    )
    embedder = await resource_manager.get_embedder(config.embedder)
    reranker = await resource_manager.get_reranker(config.reranker)
    return LongTermMemoryParams(
        session_id=config.session_id,
        vector_graph_store=vector_graph_store,
        embedder=embedder,
        reranker=reranker,
    )
