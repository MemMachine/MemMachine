import asyncio
import json
import os
from importlib import import_module

import pytest
import pytest_asyncio
from memmachine.semantic_memory.semantic_prompt import SemanticPrompt

from memmachine.common.embedder.openai_embedder import OpenAIEmbedder
from memmachine.common.language_model.openai_language_model import OpenAILanguageModel
from memmachine.semantic_memory.semantic_memory import (
    SemanticMemoryManagerParams,
    SemanticService,
)


@pytest.fixture
def config():
    open_api_key = os.environ.get("OPENAI_API_KEY")
    return {
        "api_key": open_api_key,
        "llm_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "prompt_module": "profile_prompt",
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


@pytest.fixture
def storage(asyncpg_profile_storage):
    return asyncpg_profile_storage


@pytest.fixture
def prompt(config):
    prompt_module = import_module(
        f"memmachine.server.prompt.{config['prompt_module']}", __package__
    )
    return SemanticPrompt.load_from_module(prompt_module)


@pytest_asyncio.fixture
async def semantic_memory(
    embedder,
    llm_model,
    prompt,
    storage,
):
    mem = SemanticService(
        SemanticMemoryManagerParams(
            model=llm_model,
            embeddings=embedder,
            prompt=prompt,
            semantic_storage=storage,
        )
    )
    yield mem
    await mem.delete_all()
    await mem.stop()


class TestLongMemEvalIngestion:
    @staticmethod
    async def ingest_question_convos(
        user_id: str,
        semantic_memory: SemanticService,
        conversation_sessions: list[list[dict[str, str]]],
    ):
        for convo in conversation_sessions:
            for turn in convo:
                await semantic_memory.add_persona_message(
                    set_id=user_id,
                    content=turn["content"],
                )

    @staticmethod
    async def eval_answer(
        user_id: str,
        semantic_memory: SemanticService,
        question_str: str,
        llm_model: OpenAILanguageModel,
    ):
        semantic_search_resp = await semantic_memory.semantic_search(
            query=question_str, set_id=user_id
        )
        semantic_search_resp = semantic_search_resp[:4]

        system_prompt = (
            "You are an AI assistant who answers questions based on provided information. "
            "I will give you the user's features and a conversation between a user and an assistnat. "
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
        semantic_memory,
        llm_model,
    ):
        await self.ingest_question_convos(
            "test_user",
            semantic_memory=semantic_memory,
            conversation_sessions=long_mem_convos,
        )
        count = 1
        for i in range(1200):
            count = await semantic_memory.uningested_message_count()
            if count == 0:
                break
            await asyncio.sleep(1)

        if count != 0:
            pytest.fail("Messages are not ingested")

        eval_resp = await self.eval_answer(
            "test_user",
            semantic_memory=semantic_memory,
            question_str=long_mem_question,
            llm_model=llm_model,
        )

        assert (
            "The relevant information is not found in the provided context"
            not in eval_resp
        )
