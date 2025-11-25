"""Integration test for top-level :class:`MemMachine`."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pytest
import pytest_asyncio

from memmachine import setup_nltk
from memmachine.common.configuration import (
    Configuration,
    EpisodeStoreConf,
    LogConf,
    PromptConf,
    ResourcesConf,
    SemanticMemoryConf,
    SessionManagerConf,
)
from memmachine.common.configuration.episodic_config import (
    EpisodicMemoryConfPartial,
    LongTermMemoryConfPartial,
    ShortTermMemoryConfPartial,
)
from memmachine.episode_store.episode_model import EpisodeEntry
from memmachine.main.memmachine import MemMachine, MemoryType
from memmachine.semantic_memory.semantic_model import SetIdT


@pytest.fixture
def session_data():
    @dataclass
    class _SessionData:
        user_profile_id: SetIdT | None
        session_id: SetIdT | None
        role_profile_id: SetIdT | None
        session_key: str | None

    return _SessionData(
        user_profile_id="test_user",
        session_id="test_session",
        session_key="test_session",
        role_profile_id=None,
    )


@pytest.fixture(scope="session")
def long_mem_data():
    data_path = Path("tests/data/longmemeval_snippet.json")
    with data_path.open("r", encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(scope="session")
def long_mem_question(long_mem_data):
    return long_mem_data["question"]


@pytest.fixture(scope="session")
def long_mem_conversations(long_mem_data):
    return long_mem_data["haystack_sessions"]


@pytest.fixture(scope="session", autouse=True)
def session_setup():
    setup_nltk()


@pytest_asyncio.fixture
async def memmachine_config(openai_integration_config, pg_server, neo4j_container):
    neo4j_uri = urlparse(neo4j_container["uri"])

    resources_config = {
        "embedders": {
            "openai_embedder": {
                "provider": "openai",
                "config": {
                    "model": openai_integration_config["embedding_model"],
                    "api_key": openai_integration_config["api_key"],
                    "dimensions": 1536,
                },
            }
        },
        "language_models": {
            "openai_model": {
                "provider": "openai-responses",
                "config": {
                    "model": openai_integration_config["llm_model"],
                    "api_key": openai_integration_config["api_key"],
                },
            }
        },
        "rerankers": {
            "identity_reranker": {
                "provider": "identity",
            },
        },
        "databases": {
            "pg_store": {
                "provider": "postgres",
                "config": {
                    "host": pg_server["host"],
                    "port": pg_server["port"],
                    "user": pg_server["user"],
                    "password": pg_server["password"],
                    "db_name": pg_server["database"],
                },
            },
            "neo4j_store": {
                "provider": "neo4j",
                "config": {
                    "host": neo4j_uri.hostname,
                    "port": neo4j_uri.port,
                    "user": neo4j_container["username"],
                    "password": neo4j_container["password"],
                },
            },
        },
    }

    episodic_conf = EpisodicMemoryConfPartial(
        long_term_memory=LongTermMemoryConfPartial(
            vector_graph_store="neo4j_store",
            embedder="openai_embedder",
            reranker="identity_reranker",
        ),
        short_term_memory=ShortTermMemoryConfPartial(
            llm_model="openai_model",
        ),
    )

    conf = Configuration(
        episodic_memory=episodic_conf,
        semantic_memory=SemanticMemoryConf(
            database="pg_store",
            llm_model="openai_model",
            embedding_model="openai_embedder",
        ),
        logging=LogConf(),
        prompt=PromptConf(),
        session_manager=SessionManagerConf(database="pg_store"),
        resources=ResourcesConf(**resources_config),
        episode_store=EpisodeStoreConf(database="pg_store"),
    )

    return conf


class TestMemMachineLongMemEval:
    @staticmethod
    async def _ingest_conversations(
        memmachine: MemMachine,
        session_data,
        conversations,
    ) -> None:
        for convo in conversations:
            for turn in convo:
                await memmachine.add_episodes(
                    session_data,
                    [
                        EpisodeEntry(
                            content=turn["content"],
                            producer_id="profile_id",
                            producer_role=turn.get("role", "user"),
                        )
                    ],
                )

    @staticmethod
    async def _wait_for_semantic_features(
        memmachine: MemMachine, session_data, *, timeout_seconds: int = 1200
    ) -> None:
        """Poll via the public list API until semantic memory finishes ingestion."""

        for _ in range(timeout_seconds):
            list_result = await memmachine.list_search(
                session_data,
                target_memories=[MemoryType.Semantic],
                limit=1,
            )
            if list_result.semantic_memory:
                return

            await asyncio.sleep(1)

        pytest.fail("Messages were not ingested by semantic memory")

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.integration
    async def test_long_mem_eval_via_memmachine(
        self,
        memmachine_config: Configuration,
        long_mem_conversations,
        long_mem_question,
        llm_model,
        session_data,
    ) -> None:
        memmachine = MemMachine(memmachine_config)
        await memmachine.start()
        await memmachine.create_session(session_data.session_key)

        try:
            await self._ingest_conversations(
                memmachine,
                session_data,
                long_mem_conversations,
            )

            await self._wait_for_semantic_features(memmachine, session_data)

            result = await memmachine.query_search(
                session_data,
                target_memories=[MemoryType.Semantic, MemoryType.Episodic],
                query=long_mem_question,
            )
            assert result.semantic_memory, "Semantic memory returned no features"
            assert result.episodic_memory is not None
            assert result.episodic_memory.long_term_memory
            assert result.episodic_memory.short_term_memory

            semantic_features = (result.semantic_memory or [])[:4]
            episodic_context = [
                *result.episodic_memory.long_term_memory[:4],
                *result.episodic_memory.short_term_memory[:4],
            ]

            system_prompt = (
                "You are an AI assistant who answers questions based on provided information. "
                "I will give you the user's features and a conversation between a user and an assistant. "
                "Please answer the question based on the relevant history context and user's information. "
                "If relevant information is not found, please say that you don't know with the exact format: "
                "'The relevant information is not found in the provided context.'"
            )

            episodic_prompt = "\n".join(
                f"- {episode.content}"
                for episode in episodic_context
                if episode.content
            )
            eval_prompt = (
                "Persona Profile:\n"
                f"{semantic_features}\n"
                "Episode Context:\n"
                f"{episodic_prompt}\n"
                f"Question: {long_mem_question}\nAnswer:"
            )
            eval_resp = await llm_model.generate_response(system_prompt, eval_prompt)

            assert (
                "The relevant information is not found in the provided context"
                not in eval_resp
            ), eval_resp
        finally:
            await memmachine.stop()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_memmachine_smoke_ingests_all_memories(
        self,
        memmachine_config: Configuration,
        session_data,
        long_mem_conversations,
    ) -> None:
        memmachine = MemMachine(memmachine_config)
        await memmachine.start()
        await memmachine.create_session(session_data.session_key)

        semantic_service = await memmachine._resources.get_semantic_service()
        semantic_service._feature_update_message_limit = 0

        smoke_convo = list(long_mem_conversations[0])
        if len(smoke_convo) > 2:
            smoke_convo = smoke_convo[:2]

        try:
            await self._ingest_conversations(
                memmachine,
                session_data,
                [smoke_convo],
            )

            await self._wait_for_semantic_features(
                memmachine, session_data, timeout_seconds=120
            )

            list_result = await memmachine.list_search(
                session_data,
                target_memories=[MemoryType.Semantic, MemoryType.Episodic],
            )

            assert list_result.semantic_memory, "Semantic memory returned no features"
            assert len(list_result.semantic_memory) > 0
            assert list_result.episodic_memory is not None
            assert len(list_result.episodic_memory) > 0
        finally:
            await memmachine.stop()
