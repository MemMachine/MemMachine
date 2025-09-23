"""
A related episode postulator implementation
that postulates related previous episodes.

This is suitable for use cases
where recent episodes are likely to be relevant to the current episode.
"""

import json
from datetime import datetime
from typing import Any, cast

from memmachine.common.vector_graph_store import VectorGraphStore

from ..data_types import (
    ContentType,
    Episode,
    FilterablePropertyValue,
    demangle_filterable_property_key,
    is_mangled_filterable_property_key,
    mangle_filterable_property_key,
)
from .related_episode_postulator import RelatedEpisodePostulator


class PreviousRelatedEpisodePostulator(RelatedEpisodePostulator):
    """
    RelatedEpisodePostulator implementation
    that postulates related previous episodes.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize a PreviousRelatedEpisodePostulator
        with the provided configuration.

        Args:
            config (dict[str, Any]):
                Configuration dictionary containing:
                - vector_graph_store (VectorGraphStore):
                  An instance of a VectorGraphStore
                  to use for searching episodes.
                - search_limit (int, optional):
                  The maximum number of related previous episodes
                  to postulate (default: 1).
                - filterable_property_keys (set[str], optional):
                  A set of property keys
                  to use for filtering episodes.
                  If not provided, no filtering is applied.

        Raises:
            ValueError:
                If configuration argument values are missing or invalid.
            TypeError:
                If configuration argument values are of incorrect type.
        """
        vector_graph_store = config.get("vector_graph_store")
        if vector_graph_store is None:
            raise ValueError("Vector graph store must be provided")
        if not isinstance(vector_graph_store, VectorGraphStore):
            raise TypeError(
                "Vector graph store must be an instance of VectorGraphStore"
            )

        self._vector_graph_store = vector_graph_store

        self._search_limit = config.get("search_limit", 1)

        self._filterable_property_keys = (
            config.get("filterable_property_keys") or set()
        )

    async def postulate(self, episode: Episode) -> list[Episode]:
        previous_episode_nodes = (
            await self._vector_graph_store.search_directional_nodes(
                by_property="timestamp",
                start_at_value=episode.timestamp,
                order_ascending=False,
                limit=self._search_limit,
                required_labels={"Episode"},
                required_properties={
                    mangle_filterable_property_key(
                        key
                    ): episode.filterable_properties[key]
                    for key in self._filterable_property_keys
                    if key in episode.filterable_properties
                },
            )
        )

        previous_episodes = [
            Episode(
                uuid=previous_episode_node.uuid,
                episode_type=cast(
                    str,
                    previous_episode_node.properties["episode_type"],
                ),
                content_type=ContentType(
                    previous_episode_node.properties["content_type"]
                ),
                content=previous_episode_node.properties["content"],
                timestamp=cast(
                    datetime,
                    previous_episode_node.properties.get(
                        "timestamp", datetime.min
                    ),
                ),
                filterable_properties={
                    demangle_filterable_property_key(key): cast(
                        FilterablePropertyValue, value
                    )
                    for key, value in previous_episode_node.properties.items()
                    if is_mangled_filterable_property_key(key)
                },
                user_metadata=json.loads(
                    cast(
                        str,
                        previous_episode_node.properties["user_metadata"],
                    )
                ),
            )
            for previous_episode_node in previous_episode_nodes
        ]

        return previous_episodes
