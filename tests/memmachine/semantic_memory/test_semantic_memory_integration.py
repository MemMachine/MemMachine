import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import pytest_asyncio

from memmachine.common.episode_store import EpisodeEntry, EpisodeStorage
from memmachine.common.language_model.openai_responses_language_model import (
    OpenAIResponsesLanguageModel,
)
from memmachine.semantic_memory.semantic_memory import (
    SemanticService,
)
from memmachine.semantic_memory.semantic_model import (
    ResourceRetriever,
    Resources,
    SetIdT,
)
from memmachine.semantic_memory.semantic_session_manager import (
    IsolationType,
    SemanticSessionManager,
)
from memmachine.server.prompt.profile_prompt import UserProfileSemanticCategory
from tests.memmachine.semantic_memory.mock_semantic_memory_objects import (
    SimpleSessionResourceRetriever,
)


@pytest.fixture
def embedder(openai_embedder):
    return openai_embedder


@pytest.fixture
def llm_model(real_llm_model):
    return real_llm_model


@pytest_asyncio.fixture
async def storage(pgvector_semantic_storage):
    yield pgvector_semantic_storage
    await pgvector_semantic_storage.delete_all()


@pytest.fixture
def session_types():
    return [
        UserProfileSemanticCategory,
    ]


@pytest.fixture
def profile_types():
    return [
        UserProfileSemanticCategory,
    ]


@pytest.fixture
def default_session_resources(
    llm_model,
    embedder,
    session_types,
    profile_types,
):
    return {
        IsolationType.SESSION: Resources(
            embedder=embedder,
            language_model=llm_model,
            semantic_categories=session_types,
        ),
        IsolationType.ROLE: Resources(
            embedder=embedder,
            language_model=llm_model,
            semantic_categories=[],
        ),
        IsolationType.USER: Resources(
            embedder=embedder,
            language_model=llm_model,
            semantic_categories=profile_types,
        ),
    }


@pytest.fixture
def resource_retriever(
    default_session_resources: dict[IsolationType, Resources],
):
    r = SimpleSessionResourceRetriever(
        default_resources=default_session_resources,
    )
    assert isinstance(r, ResourceRetriever)
    return r


@pytest.fixture
def basic_session_data():
    @dataclass
    class _SessionData:
        user_profile_id: SetIdT | None
        session_id: SetIdT | None
        role_profile_id: SetIdT | None

    return _SessionData(
        user_profile_id="test_user",
        session_id="test_session",
        role_profile_id=None,
    )


@pytest_asyncio.fixture
async def semantic_service(
    storage,
    episode_storage: EpisodeStorage,
    resource_retriever: ResourceRetriever,
):
    mem = SemanticService(
        SemanticService.Params(
            semantic_storage=storage,
            episode_storage=episode_storage,
            resource_retriever=resource_retriever,
            feature_update_interval_sec=0.05,
            feature_update_message_limit=0,
            debug_fail_loudly=True,
        ),
    )
    await mem.start()
    yield mem
    await mem.stop()


@pytest_asyncio.fixture
async def semantic_memory(
    semantic_service: SemanticService,
):
    return SemanticSessionManager(
        semantic_service=semantic_service,
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
                h_id = episodes[0].uid

                await semantic_memory.add_message(
                    session_data=session_data,
                    episode_ids=[h_id],
                )

    @staticmethod
    async def eval_answer(
        session_data: SemanticSessionManager.SessionData,
        semantic_memory: SemanticSessionManager,
        question_str: str,
        llm_model: OpenAIResponsesLanguageModel,
    ):
        semantic_search_resp = await semantic_memory.search(
            message=question_str,
            session_data=session_data,
        )
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

        memories = await semantic_memory.get_set_features(
            session_data=basic_session_data,
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
