import asyncio
import json
import os
from importlib import import_module

import pytest
import pytest_asyncio
from testcontainers.postgres import PostgresContainer

from memmachine.common.embedder.openai_embedder import OpenAIEmbedder
from memmachine.common.language_model.openai_language_model import OpenAILanguageModel
from memmachine.profile_memory.profile_memory import ProfileMemory
from memmachine.profile_memory.prompt_provider import ProfilePrompt
from memmachine.profile_memory.storage.asyncpg_profile import AsyncPgProfileStorage
from memmachine.profile_memory.storage.syncschema import sync_to as setup_pg_schema


@pytest.fixture
def config():
    open_api_key = os.environ.get("OPENAI_API_KEY")
    return {
        "api_key": open_api_key,
        "llm_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "prompt_module": "writing_assistant_prompt",
    }


@pytest.fixture
def embedder(config):
    return OpenAIEmbedder(
        {"api_key": config["api_key"], "model": config["embedding_model"]}
    )


@pytest.fixture
def llm_model(config):
    return OpenAILanguageModel(
        {"api_key": config["api_key"], "model": config["llm_model"]}
    )


@pytest.fixture(scope="session")
def pg_container():
    with PostgresContainer("pgvector/pgvector:pg16") as container:
        yield container


@pytest_asyncio.fixture(scope="session")
async def pg_server(pg_container):
    host = pg_container.get_container_host_ip()
    port = int(pg_container.get_exposed_port(5432))
    database = pg_container.dbname
    user = pg_container.username
    password = pg_container.password

    await setup_pg_schema(
        database=database,
        host=host,
        port=f"{port}",
        user=user,
        password=password,
    )

    yield {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "database": database,
    }


@pytest.fixture
def asyncpg_profile_storage(pg_server):
    storage = AsyncPgProfileStorage(pg_server)
    yield storage


@pytest.fixture
def storage(asyncpg_profile_storage):
    return asyncpg_profile_storage


@pytest.fixture
def prompt(config):
    prompt_module = import_module(
        f"memmachine.server.prompt.{config['prompt_module']}", __package__
    )
    return ProfilePrompt.load_from_module(prompt_module)


@pytest_asyncio.fixture
async def profile_memory(
    embedder,
    llm_model,
    prompt,
    storage,
):
    mem = ProfileMemory(
        model=llm_model,
        embeddings=embedder,
        prompt=prompt,
        profile_storage=storage,
    )
    await mem.startup()
    yield mem
    await mem.delete_all()
    await mem.cleanup()


class TestLongMemEvalIngestion:
    @staticmethod
    async def ingest_question_convos(
        user_id: str,
        profile_memory: ProfileMemory,
        conversation_sessions: list[list[dict[str, str]]],
    ):
        for convo in conversation_sessions:
            for turn in convo:
                await profile_memory.add_persona_message(
                    user_id=user_id,
                    content=turn["content"],
                )

    @staticmethod
    async def eval_answer(
        user_id: str,
        profile_memory: ProfileMemory,
        question_str: str,
        llm_model: OpenAILanguageModel,
    ):
        profile_search_resp = await profile_memory.semantic_search(
            question_str, user_id=user_id
        )
        profile_search_resp = profile_search_resp[:4]

        system_prompt = (
            "You are an AI assistant who answers questions based on provided information. "
            "I will give you the user persona profile between a user and an assistnat. "
            "Please answer the question based on the relevant history context and user's persona profile information. "
            "If relevant information is not found, please say that you don't know with the exact format: "
            "'The relevant information is not found in the user's persona profile.'."
        )

        answer_prompt_template = "Persona Profile:\n{}\nQuestion: {}\nAnswer:"

        eval_prompt = answer_prompt_template.format(profile_search_resp, question_str)
        eval_resp = await llm_model.generate_response(system_prompt, eval_prompt)
        return eval_resp

    @pytest.fixture
    def long_mem_raw_question(self):
        with open("tests/data/longmemeval_snippet.json", "r", encoding="utf-8") as f:
            data = json.load(f)
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
    async def test_periodic_mem_eval(
        self,
        long_mem_convos,
        long_mem_question,
        long_mem_answer,
        profile_memory,
        llm_model,
    ):
        await self.ingest_question_convos(
            "test_user",
            profile_memory=profile_memory,
            conversation_sessions=long_mem_convos,
        )
        count = 1
        for i in range(1200):
            count = await profile_memory.uningested_message_count()
            if count == 0:
                break
            await asyncio.sleep(1)

        if count != 0:
            pytest.fail("Messages are not ingested")

        eval_resp = await self.eval_answer(
            "test_user",
            profile_memory=profile_memory,
            question_str=long_mem_question,
            llm_model=llm_model,
        )

        assert (
            "The relevant information is not found in the user's persona profile"
            not in eval_resp
        )
