import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
import pytest_asyncio

from memmachine.common.embedder import Embedder
from memmachine.common.episode_store import EpisodeEntry, EpisodeStorage
from memmachine.common.language_model import LanguageModel
from memmachine.common.language_model.openai_responses_language_model import (
    OpenAIResponsesLanguageModel,
)
from memmachine.semantic_memory.config_store.config_store import SemanticConfigStorage
from memmachine.semantic_memory.semantic_memory import (
    ResourceManager,
    SemanticService,
)
from memmachine.semantic_memory.semantic_model import (
    SetIdT,
)
from memmachine.semantic_memory.semantic_session_manager import (
    SemanticSessionManager,
)
from memmachine.semantic_memory.storage.storage_base import SemanticStorage
from memmachine.server.prompt.coding_style_prompt import CodingStyleSemanticCategory
from memmachine.server.prompt.profile_prompt import UserProfileSemanticCategory


async def _collect_async(iterator):
    return [item async for item in iterator]


@pytest.fixture
def embedder(openai_embedder):
    return openai_embedder


@pytest.fixture
def llm_model(real_llm_model):
    return real_llm_model


@pytest.fixture
def session_types():
    return [
        UserProfileSemanticCategory,
        CodingStyleSemanticCategory,
    ]


@pytest.fixture
def profile_types():
    return [
        UserProfileSemanticCategory,
    ]


@pytest.fixture
def default_session_categories(
    session_types,
    profile_types,
):
    s_categories = {
        SemanticSessionManager.SetType.OrgSet: session_types,
        SemanticSessionManager.SetType.ProjectSet: profile_types,
        SemanticSessionManager.SetType.UserSet: session_types,
        SemanticSessionManager.SetType.OtherSet: [],
    }

    def get_cats(set_id: SetIdT):
        isolation_type = SemanticSessionManager.get_default_set_id_type(set_id)
        return s_categories[isolation_type]

    return get_cats


@pytest.fixture
def basic_session_data():
    @dataclass
    class _SessionData:
        org_id: str
        project_id: str

    return _SessionData(
        org_id="test_org",
        project_id="test_project",
    )


@pytest_asyncio.fixture
async def semantic_service(
    episode_storage: EpisodeStorage,
    semantic_storage: SemanticStorage,
    semantic_config_storage: SemanticConfigStorage,
    embedder: Embedder,
    llm_model: LanguageModel,
    default_session_categories,
):
    class _ResourceManager:
        async def get_embedder(self, _: str) -> Embedder:
            return embedder

        async def get_language_model(self, _: str) -> LanguageModel:
            return llm_model

    mem = SemanticService(
        SemanticService.Params(
            semantic_storage=semantic_storage,
            semantic_config_storage=semantic_config_storage,
            episode_storage=episode_storage,
            feature_update_interval_sec=0.05,
            uningested_message_limit=0,
            debug_fail_loudly=True,
            default_embedder=embedder,
            default_embedder_name="default_embedder",
            default_language_model=llm_model,
            default_category_retriever=default_session_categories,
            resource_manager=cast(ResourceManager, _ResourceManager()),
        ),
    )
    await mem.start()
    yield mem
    await mem.stop()


@pytest_asyncio.fixture
async def semantic_memory(
    semantic_service: SemanticService,
    semantic_config_storage: SemanticConfigStorage,
):
    return SemanticSessionManager(
        semantic_service=semantic_service,
        semantic_config_storage=semantic_config_storage,
    )


class TestLongMemEvalIngestion:
    @staticmethod
    async def ingest_question_convos(
        session_data: SemanticSessionManager.SessionData,
        semantic_memory: SemanticSessionManager,
        history_storage: EpisodeStorage,
        conversation_sessions: list[list[dict[str, str]]],
    ):
        for convo in conversation_sessions:
            for turn in convo:
                episodes = await history_storage.add_episodes(
                    episodes=[
                        EpisodeEntry(
                            content=turn["content"],
                            producer_id="profile_id",
                            producer_role="dev",
                        ),
                    ],
                    session_key="session_id",
                )

                assert len(episodes) == 1

                await semantic_memory.add_message(
                    session_data=session_data,
                    episodes=episodes,
                )

    @staticmethod
    async def eval_answer(
        session_data: SemanticSessionManager.SessionData,
        semantic_memory: SemanticSessionManager,
        question_str: str,
        llm_model: OpenAIResponsesLanguageModel,
    ):
        semantic_search_resp = [
            result
            async for result in semantic_memory.search(
                message=question_str,
                session_data=session_data,
            )
        ]
        semantic_search_resp = semantic_search_resp[:4]

        system_prompt = (
            "You are an AI assistant who answers questions based on provided information. "
            "I will give you the user's features and a conversation between a user and an assistant. "
            "Please answer the question based on the relevant history context and user's information. "
            "If relevant information is not found, please say that you don't know with the exact format: "
            "'The relevant information is not found in the provided context.'."
        )

        answer_prompt_template = "Persona Profile:\n{}\nQuestion: {}\nAnswer:"

        eval_prompt = answer_prompt_template.format(semantic_search_resp, question_str)
        eval_resp = await llm_model.generate_response(system_prompt, eval_prompt)
        return eval_resp

    @pytest.fixture
    def long_mem_raw_question(self):
        data_path = Path("tests/data/longmemeval_snippet.json")
        with data_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data

    @pytest.fixture
    def long_mem_convos(self, long_mem_raw_question):
        return long_mem_raw_question["haystack_sessions"]

    @pytest.fixture
    def long_mem_question(self, long_mem_raw_question):
        return long_mem_raw_question["question"]

    @pytest.fixture
    def long_mem_answer(self, long_mem_raw_question):
        return long_mem_raw_question["answer"]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_long_mem_eval_smoke(
        self,
        semantic_memory,
        episode_storage: EpisodeStorage,
        basic_session_data,
        long_mem_convos,
    ):
        smoke_convos = long_mem_convos[0]
        if len(smoke_convos) > 2:
            smoke_convos = smoke_convos[:2]

        await self.ingest_question_convos(
            basic_session_data,
            semantic_memory=semantic_memory,
            history_storage=episode_storage,
            conversation_sessions=[smoke_convos],
        )
        count = 1
        for _i in range(60):
            count = await semantic_memory.number_of_uningested_messages(
                session_data=basic_session_data,
            )

            if count == 0:
                break
            await asyncio.sleep(1)

        if count != 0:
            pytest.fail(f"Messages are not ingested, count={count}")

        memories = await _collect_async(
            semantic_memory.get_set_features(
                session_data=basic_session_data,
            )
        )
        assert len(memories) > 0

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.integration
    async def test_periodic_mem_eval(
        self,
        long_mem_convos,
        long_mem_question,
        long_mem_answer,
        semantic_memory,
        episode_storage: EpisodeStorage,
        llm_model,
        basic_session_data,
    ):
        await self.ingest_question_convos(
            basic_session_data,
            semantic_memory=semantic_memory,
            history_storage=episode_storage,
            conversation_sessions=long_mem_convos,
        )
        count = 1
        for _i in range(1200):
            count = await semantic_memory.number_of_uningested_messages(
                session_data=basic_session_data,
            )

            if count == 0:
                break
            await asyncio.sleep(1)

        if count != 0:
            pytest.fail(f"Messages are not ingested, count={count}")

        eval_resp = await self.eval_answer(
            session_data=basic_session_data,
            semantic_memory=semantic_memory,
            question_str=long_mem_question,
            llm_model=llm_model,
        )

        assert (
            "The relevant information is not found in the provided context"
            not in eval_resp
        )
