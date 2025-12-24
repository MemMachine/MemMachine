"""Helpers for constructing short-term memory instances."""

from pydantic import InstanceOf

from memmachine.common.configuration.episodic_config import ShortTermMemoryConf
from memmachine.common.resource_manager import CommonResourceManager

from .short_term_memory import ShortTermMemoryParams


async def short_term_memory_params_from_config(
    config: ShortTermMemoryConf,
    resource_manager: InstanceOf[CommonResourceManager],
) -> ShortTermMemoryParams:
    """Create ShortTermMemoryParams from configuration and common resources."""
    session_data_manager = await resource_manager.get_session_data_manager()
    return ShortTermMemoryParams(
        session_key=config.session_key,
        llm_model=await resource_manager.get_language_model(
            config.llm_model, validate=True
        ),
        data_manager=session_data_manager,
        summary_prompt_system=config.summary_prompt_system,
        summary_prompt_user=config.summary_prompt_user,
        message_capacity=config.message_capacity,
    )
