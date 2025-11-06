"""
Declarative memory system for storing and retrieving
episodic and semantic memory.
"""

import asyncio
import json
import logging
from collections import deque
from collections.abc import Iterable, Mapping
from datetime import datetime
from string import Template
from typing import cast
from uuid import UUID, uuid4

from nltk import sent_tokenize
from pydantic import BaseModel, Field, InstanceOf

from memmachine.common.embedder.embedder import Embedder
from memmachine.common.reranker.reranker import Reranker
from memmachine.common.vector_graph_store import Edge, Node, VectorGraphStore

from .data_types import (
    Chunk,
    ContentType,
    Derivative,
    Episode,
    FilterablePropertyValue,
    demangle_filterable_property_key,
    is_mangled_filterable_property_key,
    mangle_filterable_property_key,
)

logger = logging.getLogger(__name__)


class DeclarativeMemoryParams(BaseModel):
    """
    Parameters for DeclarativeMemory.

    Attributes:
        session_id (str):
            Session identifier.
        max_chunk_length (int):
            Maximum length of a chunk in characters
            (default: 1000).
        vector_graph_store (VectorGraphStore):
            VectorGraphStore instance
            for storing and retrieving memories.
        embedder (Embedder):
            Embedder instance for creating embeddings.
        reranker (Reranker):
            Reranker instance for reranking search results.
    """

    session_id: str = Field(
        ...,
        description="Session identifier",
    )
    max_chunk_length: int = Field(
        1000,
        description="Maximum length of a chunk in characters.",
        gt=0,
    )
    vector_graph_store: InstanceOf[VectorGraphStore] = Field(
        ...,
        description="VectorGraphStore instance for storing and retrieving memories",
    )
    embedder: InstanceOf[Embedder] = Field(
        ...,
        description="Embedder instance for creating embeddings",
    )
    reranker: InstanceOf[Reranker] = Field(
        ...,
        description="Reranker instance for reranking search results",
    )


class DeclarativeMemory:
    """
    Declarative memory system.
    """

    def __init__(self, params: DeclarativeMemoryParams):
        """
        Initialize a DeclarativeMemory with the provided parameters.

        Args:
            params (DeclarativeMemoryParams):
                Parameters for the DeclarativeMemory.
        """
        session_id = params.session_id

        self._max_chunk_length = params.max_chunk_length
        self._vector_graph_store = params.vector_graph_store
        self._embedder = params.embedder
        self._reranker = params.reranker

        self._episode_collection = f"Episode_{session_id}"
        self._chunk_collection = f"Chunk_{session_id}"
        self._derivative_collection = f"Derivative_{session_id}"

        self._contains_relation = f"CONTAINS_{session_id}"
        self._derived_from_relation = f"DERIVED_FROM_{session_id}"

        self._message_prefix_template = Template("[$timestamp] $source:\n")
        self._message_suffix_template = Template("")
        self._temporal_context: deque[Chunk] | None = None

    async def add_episodes(
        self,
        episodes: Iterable[Episode],
    ):
        """
        Add episodes.

        Episodes are sorted by timestamp.
        Episodes with the same timestamp are sorted by UUID.

        Args:
            episodes (Iterable[Episode]): The episodes to add.
        """

        # TODO @edwinyyyu: lock this
        # The memory is server-side so it can be stateful.
        # Remember previous episodes for context.
        # This is like coming back to work on a Monday.
        if self._temporal_context is None:
            self._temporal_context = deque(maxlen=20)
        # task = self._vector_graph_store.get_last_episodes

        episodes = sorted(
            episodes, key=lambda episode: (episode.timestamp, episode.uuid)
        )
        episode_nodes = [
            Node(
                uuid=episode.uuid,
                properties={
                    "uuid": str(episode.uuid),
                    "timestamp": episode.timestamp,
                    "source": episode.source,
                    "content_type": episode.content_type.value,
                    "content": episode.content,
                    "user_metadata": json.dumps(episode.user_metadata),
                }
                | {
                    mangle_filterable_property_key(key): value
                    for key, value in episode.filterable_properties.items()
                },
            )
            for episode in episodes
        ]

        chunk_episode_tasks = [self._chunk_episode(episode) for episode in episodes]
        episodes_chunks = await asyncio.gather(*chunk_episode_tasks)

        chunks = [
            chunk for episode_chunks in episodes_chunks for chunk in episode_chunks
        ]

        chunk_nodes = [
            Node(
                uuid=chunk.uuid,
                properties={
                    "uuid": str(chunk.uuid),
                    "episode_uuid": str(chunk.episode_uuid),
                    "sequence_number": chunk.sequence_number,
                    "timestamp": chunk.timestamp,
                    "source": chunk.source,
                    "content_type": chunk.content_type.value,
                    "content": chunk.content,
                    "user_metadata": json.dumps(chunk.user_metadata),
                }
                | {
                    mangle_filterable_property_key(key): value
                    for key, value in chunk.filterable_properties.items()
                },
            )
            for chunk in chunks
        ]

        episode_chunk_edges = [
            Edge(
                uuid=uuid4(),
                source_uuid=episode.uuid,
                target_uuid=chunk.uuid,
            )
            for episode, episode_chunks in zip(episodes, episodes_chunks)
            for chunk in episode_chunks
        ]

        derive_derivatives_tasks = []
        for chunk in chunks:
            derive_derivatives_tasks.append(
                self._derive_derivatives(
                    chunk,
                    list(self._temporal_context),
                )
            )
            self._temporal_context.append(chunk)

        chunks_derivatives = await asyncio.gather(*derive_derivatives_tasks)

        derivatives = [
            derivative
            for chunk_derivatives in chunks_derivatives
            for derivative in chunk_derivatives
        ]

        derivative_embeddings = await self._embedder.ingest_embed(
            [derivative.content for derivative in derivatives],
        )

        derivative_nodes = [
            Node(
                uuid=derivative.uuid,
                properties={
                    "uuid": str(derivative.uuid),
                    "content_type": derivative.content_type.value,
                    "content": derivative.content,
                },
                embeddings={
                    DeclarativeMemory._embedding_name(
                        self._embedder.model_id,
                        self._embedder.dimensions,
                    ): (embedding, self._embedder.similarity_metric),
                },
            )
            for derivative, embedding in zip(derivatives, derivative_embeddings)
        ]

        derivative_chunk_edges = [
            Edge(
                uuid=uuid4(),
                source_uuid=derivative.uuid,
                target_uuid=chunk.uuid,
            )
            for chunk, chunk_derivatives in zip(chunks, chunks_derivatives)
            for derivative in chunk_derivatives
        ]

        await self._vector_graph_store.add_nodes(
            collection=self._episode_collection,
            nodes=episode_nodes,
        )
        await self._vector_graph_store.add_nodes(
            collection=self._chunk_collection,
            nodes=chunk_nodes,
        )
        await self._vector_graph_store.add_edges(
            relation=self._contains_relation,
            source_collection=self._episode_collection,
            target_collection=self._chunk_collection,
            edges=episode_chunk_edges,
        )
        await self._vector_graph_store.add_nodes(
            collection=self._derivative_collection,
            nodes=derivative_nodes,
        )
        await self._vector_graph_store.add_edges(
            relation=self._derived_from_relation,
            source_collection=self._derivative_collection,
            target_collection=self._chunk_collection,
            edges=derivative_chunk_edges,
        )

    async def _chunk_episode(self, episode: Episode) -> list[Chunk]:
        """
        Partition episode into chunks.

        Args:
            episode (Episode): The episode whose content to partition.

        Returns:
            list[Chunk]: A list of non-overlapping chunks.
        """
        match episode.content_type:
            case ContentType.MESSAGE | ContentType.TEXT:
                sentences = []

                for line in episode.content.strip().splitlines():
                    for sentence in sent_tokenize(line.strip()):
                        if len(sentence) > self._max_chunk_length:
                            sentence_splits = self._split_sentence(sentence)
                            sentences.extend(sentence_splits)
                        else:
                            sentences.append(sentence)

                return [
                    Chunk(
                        uuid=uuid4(),
                        episode_uuid=episode.uuid,
                        sequence_number=index,
                        timestamp=episode.timestamp,
                        source=episode.source,
                        content_type=episode.content_type,
                        content=sentence,
                        filterable_properties=episode.filterable_properties,
                        user_metadata=episode.user_metadata,
                    )
                    for index, sentence in enumerate(sentences)
                ]
            case _:
                return [
                    Chunk(
                        uuid=uuid4(),
                        episode_uuid=episode.uuid,
                        sequence_number=0,
                        timestamp=episode.timestamp,
                        source=episode.source,
                        content_type=episode.content_type,
                        content=episode.content,
                        filterable_properties=episode.filterable_properties,
                        user_metadata=episode.user_metadata,
                    )
                ]

    def _split_sentence(self, sentence: str) -> list[str]:
        """
        Split a long sentence into smaller chunks.

        Args:
            sentence (str): The sentence to split.

        Returns:
            list[str]: A list of sentence chunks.
        """
        return []  # TODO @edwinyyyu: Implement sentence splitting logic here.

    async def _derive_derivatives(
        self,
        chunk: Chunk,
        temporal_context: Iterable[Chunk],
    ) -> list[Derivative]:
        """
        Derive derivatives from a chunk from the temporal context.

        Args:
            chunk (Chunk):
                The chunk from which to derive derivatives.
            temporal_context (Iterable[Chunk]):
                The temporal context containing previous chunks.

        Returns:
            list[Derivative]: A list of derived derivatives.
        """
        match chunk.content_type:
            case ContentType.MESSAGE:
                message_content = (
                    DeclarativeMemory._apply_string_template(
                        self._message_prefix_template, chunk
                    )
                    + chunk.content
                    + DeclarativeMemory._apply_string_template(
                        self._message_suffix_template, chunk
                    )
                )

                return [
                    Derivative(
                        uuid=uuid4(),
                        content_type=ContentType.MESSAGE,
                        content=message_content,
                        filterable_properties=chunk.filterable_properties,
                    ),
                ]
            case ContentType.TEXT:
                text_content = chunk.content
                return [
                    Derivative(
                        uuid=uuid4(),
                        content_type=ContentType.TEXT,
                        content=text_content,
                        filterable_properties=chunk.filterable_properties,
                    )
                ]
            case _:
                return []

    async def search(
        self,
        query: str,
        max_num_chunks: int = 20,
        property_filter: Mapping[str, FilterablePropertyValue] | None = None,
    ) -> list[Chunk]:
        """
        Search declarative memory for chunks relevant to the query.

        Args:
            query (str):
                The search query.
            max_num_chunks (int):
                The maximum number of chunks to return
                (default: 20).
            property_filter (Mapping[str, FilterablePropertyValue] | None):
                Filterable property keys and values
                to use for filtering episodes
                (default: None).

        Returns:
            list[Chunk]:
                A list of chunks relevant to the query, ordered chronologically.
        """
        if property_filter is None:
            property_filter = {}

        query_embedding = (
            await self._embedder.search_embed(
                [query],
            )
        )[0]

        # Search graph store for vector matches.
        matched_derivative_nodes = await self._vector_graph_store.search_similar_nodes(
            collection=self._derivative_collection,
            embedding_name=(
                DeclarativeMemory._embedding_name(
                    self._embedder.model_id,
                    self._embedder.dimensions,
                )
            ),
            query_embedding=query_embedding,
            similarity_metric=self._embedder.similarity_metric,
            limit=100,
            required_properties={
                mangle_filterable_property_key(key): value
                for key, value in property_filter.items()
            },
        )

        # Get source chunks of matched derivatives.
        search_derivatives_source_chunk_nodes_tasks = [
            self._vector_graph_store.search_related_nodes(
                relation=self._derived_from_relation,
                other_collection=self._chunk_collection,
                this_collection=self._derivative_collection,
                this_node_uuid=matched_derivative_node.uuid,
                find_sources=False,
                find_targets=True,
                required_node_properties={
                    mangle_filterable_property_key(key): value
                    for key, value in property_filter.items()
                },
            )
            for matched_derivative_node in matched_derivative_nodes
        ]

        source_chunk_nodes = [
            chunk_node
            for chunk_nodes in await asyncio.gather(
                *search_derivatives_source_chunk_nodes_tasks
            )
            for chunk_node in chunk_nodes
        ]

        nuclear_chunks = [
            DeclarativeMemory._chunk_from_chunk_node(source_chunk_node)
            for source_chunk_node in source_chunk_nodes
        ]

        # Use source chunks as nuclei for contextualization.
        contextualize_chunk_tasks = [
            self._contextualize_chunk(
                nuclear_chunk,
                property_filter=property_filter,
            )
            for nuclear_chunk in nuclear_chunks
        ]

        chunk_contexts = await asyncio.gather(*contextualize_chunk_tasks)

        # Rerank chunk contexts.
        chunk_context_scores = await self._score_chunk_contexts(query, chunk_contexts)

        reranked_anchored_chunk_contexts = [
            (nuclear_chunk, chunk_context)
            for _, nuclear_chunk, chunk_context in sorted(
                zip(
                    chunk_context_scores,
                    nuclear_chunks,
                    chunk_contexts,
                ),
                key=lambda triple: triple[0],
                reverse=True,
            )
        ]

        # Unify chunk contexts.
        unified_chunk_context = DeclarativeMemory._unify_anchored_chunk_contexts(
            reranked_anchored_chunk_contexts,
            max_num_chunks=max_num_chunks,
        )
        return unified_chunk_context

    async def _contextualize_chunk(
        self,
        nuclear_chunk: Chunk,
        max_backward_chunks: int = 1,
        max_forward_chunks: int = 2,
        property_filter: Mapping[str, FilterablePropertyValue] | None = None,
    ) -> list[Chunk]:
        if property_filter is None:
            property_filter = {}

        previous_chunk_nodes = await self._vector_graph_store.search_directional_nodes(
            collection=self._chunk_collection,
            by_properties=("timestamp", "episode_uuid", "sequence_number"),
            starting_at=(
                nuclear_chunk.timestamp,
                str(nuclear_chunk.episode_uuid),
                str(nuclear_chunk.sequence_number),
            ),
            order_ascending=(False, False, False),
            include_equal_start=False,
            limit=max_backward_chunks,
            required_properties={
                mangle_filterable_property_key(key): value
                for key, value in property_filter.items()
            },
        )

        next_chunk_nodes = await self._vector_graph_store.search_directional_nodes(
            collection=self._chunk_collection,
            by_properties=("timestamp", "episode_uuid", "sequence_number"),
            starting_at=(
                nuclear_chunk.timestamp,
                str(nuclear_chunk.episode_uuid),
                str(nuclear_chunk.sequence_number),
            ),
            order_ascending=(True, True, True),
            include_equal_start=False,
            limit=max_forward_chunks,
            required_properties={
                mangle_filterable_property_key(key): value
                for key, value in property_filter.items()
            },
        )

        context = (
            [
                DeclarativeMemory._chunk_from_chunk_node(chunk_node)
                for chunk_node in reversed(previous_chunk_nodes)
            ]
            + [nuclear_chunk]
            + [
                DeclarativeMemory._chunk_from_chunk_node(chunk_node)
                for chunk_node in next_chunk_nodes
            ]
        )

        return context

    async def _score_chunk_contexts(
        self, query: str, chunk_contexts: Iterable[Iterable[Chunk]]
    ) -> list[float]:
        """
        Score chunk node contexts
        based on their relevance to the query.
        """
        context_strings = []
        for chunk_context in chunk_contexts:
            context_string = self.string_from_chunk_context(chunk_context)
            context_strings.append(context_string)

        chunk_context_scores = await self._reranker.score(query, context_strings)

        return chunk_context_scores

    async def get_episodes(self, uuids: Iterable[UUID]) -> list[Episode]:
        """
        Get episodes by their UUIDs.
        """
        episode_nodes = await self._vector_graph_store.get_nodes(
            collection=self._episode_collection,
            node_uuids=uuids,
        )

        episodes = [
            DeclarativeMemory._episode_from_episode_node(episode_node)
            for episode_node in episode_nodes
        ]

        return episodes

    async def delete_episodes(self, uuids: Iterable[UUID]):
        """
        Delete episodes by their UUIDs.
        """
        await self._vector_graph_store.delete_nodes(
            collection=self._episode_collection,
            node_uuids=uuids,
        )

    def string_from_chunk_context(self, chunk_context: Iterable[Chunk]) -> str:
        """
        Format chunk context as a string.
        """
        context_string = ""

        previous_chunk = None
        for chunk in chunk_context:
            match chunk.content_type:
                case ContentType.MESSAGE:
                    if previous_chunk is None:
                        context_string += DeclarativeMemory._apply_string_template(
                            self._message_prefix_template, chunk
                        )
                    elif chunk.episode_uuid != previous_chunk.episode_uuid:
                        context_string += (
                            DeclarativeMemory._apply_string_template(
                                self._message_suffix_template, previous_chunk
                            )
                            + "\n"
                        )
                        context_string += DeclarativeMemory._apply_string_template(
                            self._message_prefix_template, chunk
                        )

                    context_string += chunk.content + "\n"
                case ContentType.TEXT:
                    context_string += chunk.content + "\n"
                case _:
                    pass

            previous_chunk = chunk

        if previous_chunk is not None:
            match previous_chunk.content_type:
                case ContentType.MESSAGE:
                    context_string += DeclarativeMemory._apply_string_template(
                        self._message_suffix_template, previous_chunk
                    )
                case _:
                    pass

        return context_string

    @staticmethod
    def _unify_anchored_chunk_contexts(
        anchored_chunk_contexts: Iterable[tuple[Chunk, Iterable[Chunk]]],
        max_num_chunks: int,
    ) -> list[Chunk]:
        """
        Unify chunk contexts
        anchored on their nuclear chunks
        into a single list of chunks,
        respecting the chunk limit.
        """
        chunk_set: set[Chunk] = set()

        for nuclear_chunk, context in anchored_chunk_contexts:
            context = list(context)

            if len(chunk_set) >= max_num_chunks:
                break
            elif (len(chunk_set) + len(context)) <= max_num_chunks:
                # It is impossible that the context exceeds the limit.
                chunk_set.update(context)
            else:
                # It is possible that the context exceeds the limit.
                # Prioritize chunks near the nuclear chunk.
                context = list(context)

                # Sort chronological chunks by weighted index-proximity to the nuclear chunk.
                nuclear_index = context.index(nuclear_chunk)

                def weighted_index_proximity(chunk: Chunk) -> float:
                    proximity = context.index(chunk) - nuclear_index
                    if proximity >= 0:
                        return proximity / 2
                    else:
                        return -proximity

                nuclear_context = sorted(
                    context,
                    key=weighted_index_proximity,
                )

                # Add chunks to unified context until limit is reached,
                # or until the context is exhausted.
                for chunk in nuclear_context:
                    if len(chunk_set) >= max_num_chunks:
                        break
                    chunk_set.add(chunk)

        unified_chunk_context = sorted(
            chunk_set,
            key=lambda chunk: (
                chunk.timestamp,
                chunk.episode_uuid,
                chunk.sequence_number,
            ),
        )

        return unified_chunk_context

    @staticmethod
    def _apply_string_template(template: Template, data: Episode | Chunk) -> str:
        if isinstance(data, Episode) or isinstance(data, Chunk):
            return template.safe_substitute(
                {
                    "timestamp": data.timestamp,
                    "source": data.source,
                    "content": data.content,
                },
                **{
                    key: value
                    for key, value in {
                        **data.filterable_properties,
                        **(
                            data.user_metadata
                            if isinstance(data.user_metadata, dict)
                            else {}
                        ),
                    }.items()
                },
            )
        else:
            raise TypeError("Data must be an Episode or Chunk")

    @staticmethod
    def _episode_from_episode_node(episode_node: Node) -> Episode:
        return Episode(
            uuid=UUID(cast(str, episode_node.properties["uuid"])),
            timestamp=cast(datetime, episode_node.properties["timestamp"]),
            source=cast(str, episode_node.properties["source"]),
            content_type=ContentType(episode_node.properties["content_type"]),
            content=episode_node.properties["content"],
            filterable_properties={
                demangle_filterable_property_key(key): cast(
                    FilterablePropertyValue, value
                )
                for key, value in episode_node.properties.items()
                if is_mangled_filterable_property_key(key)
            },
            user_metadata=json.loads(
                cast(str, episode_node.properties["user_metadata"])
            ),
        )

    @staticmethod
    def _chunk_from_chunk_node(chunk_node: Node) -> Chunk:
        return Chunk(
            uuid=UUID(cast(str, chunk_node.properties["uuid"])),
            episode_uuid=UUID(cast(str, chunk_node.properties["episode_uuid"])),
            sequence_number=cast(int, chunk_node.properties["sequence_number"]),
            timestamp=cast(datetime, chunk_node.properties["timestamp"]),
            source=cast(str, chunk_node.properties["source"]),
            content_type=ContentType(chunk_node.properties["content_type"]),
            content=chunk_node.properties["content"],
            filterable_properties={
                demangle_filterable_property_key(key): cast(
                    FilterablePropertyValue, value
                )
                for key, value in chunk_node.properties.items()
                if is_mangled_filterable_property_key(key)
            },
            user_metadata=json.loads(cast(str, chunk_node.properties["user_metadata"])),
        )

    @staticmethod
    def _embedding_name(model_id: str, dimensions: int) -> str:
        """
        Generate a standardized property name for embeddings
        based on the model ID and embedding dimensions.

        Args:
            model_id (str): The identifier of the embedding model.
            dimensions (int): The dimensionality of the embedding.

        Returns:
            str: A standardized property name for the embedding.
        """
        return f"embedding_{model_id}_{dimensions}d"
