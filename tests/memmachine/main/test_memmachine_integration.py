"""Integration test for top-level :class:`MemMachine`."""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from pathlib import Path

import pytest

from memmachine.common.configuration import (
    Configuration,
)
from memmachine.common.episode_store import EpisodeEntry
from memmachine.main.memmachine import MemMachine, MemoryType


@pytest.fixture
def llm_model(real_llm_model):
    return real_llm_model


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
        memmachine: MemMachine,
        session_data,
        *,
        timeout_seconds: int = 1200,
    ) -> None:
        """Poll via the public list API until semantic memory finishes ingestion."""

        for _ in range(timeout_seconds):
            list_result = await memmachine.list_search(
                session_data,
                target_memories=[MemoryType.Semantic],
                page_size=1,
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
                *result.episodic_memory.long_term_memory.episodes[:4],
                *result.episodic_memory.short_term_memory.episodes[:4],
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
        memmachine: MemMachine,
        session_data,
        long_mem_conversations,
    ) -> None:
        semantic_service = await memmachine._resources.get_semantic_service()
        semantic_service._feature_update_message_limit = 0
        semantic_service._background_ingestion_interval_sec = 0.05
        semantic_service._feature_time_limit = timedelta(seconds=1)

        smoke_convo = list(long_mem_conversations[0])
        if len(smoke_convo) > 2:
            smoke_convo = smoke_convo[:2]

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

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_memmachine_list_set_ids_returns_details(
        self,
        memmachine: MemMachine,
        session_data,
    ) -> None:
        """Test that semantic_list_set_ids returns is_org_level, tags, name, and description."""
        # Create org set types with name and description
        await memmachine.create_semantic_set_type(
            session_data=session_data,
            is_org_level=True,
            metadata_tags=["user_id"],
            name="User Settings",
            description="User-specific configuration",
        )

        await memmachine.create_semantic_set_type(
            session_data=session_data,
            is_org_level=False,
            metadata_tags=["repo_id"],
            name="Repository Data",
            description="Repository-specific information",
        )

        # Create session data with metadata
        from dataclasses import dataclass

        @dataclass
        class ExtendedSessionData:
            org_id: str
            project_id: str
            metadata: dict

        extended_session = ExtendedSessionData(
            org_id=session_data.org_id,
            project_id=session_data.project_id,
            metadata={
                "user_id": "test_user",
                "repo_id": "test_repo",
            },
        )

        # Get all set IDs
        sets = await memmachine.semantic_list_set_ids(session_data=extended_session)

        # Should have 4 sets: 2 default (org + project) + 2 custom
        assert len(sets) == 4

        # Find the user set
        user_set = next((s for s in sets if "user_id" in s.tags), None)
        assert user_set is not None
        assert user_set.is_org_level is True
        assert user_set.tags == ["user_id"]
        assert user_set.name == "User Settings"
        assert user_set.description == "User-specific configuration"
        assert user_set.id is not None

        # Find the repo set
        repo_set = next((s for s in sets if "repo_id" in s.tags), None)
        assert repo_set is not None
        assert repo_set.is_org_level is False
        assert repo_set.tags == ["repo_id"]
        assert repo_set.name == "Repository Data"
        assert repo_set.description == "Repository-specific information"
        assert repo_set.id is not None

        # Verify default sets have no tags and no name/description
        default_sets = [s for s in sets if len(s.tags) == 0]
        assert len(default_sets) == 2

        # One should be org level, one should be project level
        org_level_default = next((s for s in default_sets if s.is_org_level), None)
        project_level_default = next(
            (s for s in default_sets if not s.is_org_level), None
        )
        assert org_level_default is not None
        assert project_level_default is not None
        assert org_level_default.name is None
        assert org_level_default.description is None
        assert project_level_default.name is None
        assert project_level_default.description is None
