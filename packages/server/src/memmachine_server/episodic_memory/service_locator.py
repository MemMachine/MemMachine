"""Factory helpers for wiring episodic memory components."""

from pydantic import InstanceOf

from memmachine_server.common.configuration.episodic_config import (
    EpisodicMemoryConf,
    EventLongTermMemoryConf,
)
from memmachine_server.common.resource_manager import CommonResourceManager

from .episodic_memory import EpisodicMemoryParams
from .long_term_memory.long_term_memory import LongTermMemory
from .long_term_memory.service_locator import (
    create_event_backend_partitions,
    delete_event_backend_partitions,
    long_term_memory_params_from_config,
)
from .short_term_memory.service_locator import (
    short_term_memory_params_from_config,
)
from .short_term_memory.short_term_memory import ShortTermMemory


async def episodic_memory_params_from_config(
    config: EpisodicMemoryConf,
    resource_manager: InstanceOf[CommonResourceManager],
) -> EpisodicMemoryParams:
    """Create EpisodicMemoryParams from configuration and resource manager."""
    long_term_memory: LongTermMemory | None = None
    if config.long_term_memory and config.long_term_memory_enabled:
        long_term_memory_params = await long_term_memory_params_from_config(
            config.long_term_memory,
            resource_manager,
        )
        long_term_memory = LongTermMemory(long_term_memory_params)

    short_term_memory: ShortTermMemory | None = None
    if config.short_term_memory and config.short_term_memory_enabled:
        short_term_memory_params = await short_term_memory_params_from_config(
            config.short_term_memory,
            resource_manager,
        )
        short_term_memory = await ShortTermMemory.create(short_term_memory_params)

    metrics_factory_id = config.metrics_factory_id or "prometheus"

    return EpisodicMemoryParams(
        session_key=config.session_key,
        metrics_factory=await resource_manager.get_metrics_factory(
            metrics_factory_id,
        ),
        long_term_memory=long_term_memory,
        short_term_memory=short_term_memory,
        enabled=config.enabled,
    )


async def create_episodic_memory_storage(
    config: EpisodicMemoryConf,
    resource_manager: InstanceOf[CommonResourceManager],
) -> None:
    """Create the storage a session's episodic memory writes to.

    Called once, when the session is created; no memory request creates
    storage. Only the event backend has storage of its own: the declarative
    backend writes into a shared graph store.
    """
    if not (config.long_term_memory and config.long_term_memory_enabled):
        return
    if isinstance(config.long_term_memory, EventLongTermMemoryConf):
        await create_event_backend_partitions(config.long_term_memory, resource_manager)


async def delete_episodic_memory_storage(
    config: EpisodicMemoryConf,
    resource_manager: InstanceOf[CommonResourceManager],
) -> None:
    """Delete a session's episodic memory data without opening the session.

    The event backend's partitions go by key, so a session whose storage was
    never fully created can still be deleted. The declarative backend has no
    partition of its own; its episodes are deleted through the memory.
    """
    if not (config.long_term_memory and config.long_term_memory_enabled):
        return
    if isinstance(config.long_term_memory, EventLongTermMemoryConf):
        await delete_event_backend_partitions(config.long_term_memory, resource_manager)
        return
    long_term_memory = LongTermMemory(
        await long_term_memory_params_from_config(
            config.long_term_memory, resource_manager
        )
    )
    await long_term_memory.drop_session_partition()
