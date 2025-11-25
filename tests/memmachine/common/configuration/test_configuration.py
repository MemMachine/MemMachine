from pathlib import Path

import pytest
from pydantic import SecretStr

from memmachine.common.configuration import (
    EpisodicMemoryConfPartial,
    load_config_yml_file,
)
from memmachine.common.configuration.episodic_config import (
    LongTermMemoryConfPartial,
    ShortTermMemoryConfPartial,
)
from memmachine.common.configuration.log_conf import LogLevel


@pytest.fixture
def long_term_memory_conf() -> LongTermMemoryConfPartial:
    return LongTermMemoryConfPartial(
        embedder="embedder_v1",
        reranker="reranker_v1",
        vector_graph_store="store_v1",
    )


def test_update_long_term_memory_conf(long_term_memory_conf: LongTermMemoryConfPartial):
    specific = LongTermMemoryConfPartial(
        session_id="session_123",
        embedder="embedder_v2",
    )

    updated = specific.merge(long_term_memory_conf)
    assert updated.session_id == "session_123"
    assert updated.embedder == "embedder_v2"
    assert updated.reranker == "reranker_v1"
    assert updated.vector_graph_store == "store_v1"


@pytest.fixture
def short_term_memory_conf() -> ShortTermMemoryConfPartial:
    return ShortTermMemoryConfPartial(
        llm_model="model_v1",
        message_capacity=12345,
        summary_prompt_user="Summarize the following:",
        summary_prompt_system="You are a helpful assistant.",
    )


def test_update_session_memory_conf(short_term_memory_conf: ShortTermMemoryConfPartial):
    specific = ShortTermMemoryConfPartial(
        session_key="session_123",
        message_capacity=3000,
    )

    updated = specific.merge(short_term_memory_conf)
    assert updated.session_key == "session_123"
    assert updated.llm_model == "model_v1"
    assert updated.message_capacity == 3000


def test_update_episodic_memory_conf(
    long_term_memory_conf: LongTermMemoryConfPartial,
    short_term_memory_conf: ShortTermMemoryConfPartial,
):
    base = EpisodicMemoryConfPartial(
        short_term_memory=short_term_memory_conf,
        long_term_memory=long_term_memory_conf,
        metrics_factory_id="metrics_factory_id",
    )
    specific = EpisodicMemoryConfPartial(
        session_key="session_123",
        long_term_memory=LongTermMemoryConfPartial(embedder="embedder_v2"),
    )

    updated = specific.merge(base)
    assert updated.long_term_memory.embedder == "embedder_v2"
    assert updated.long_term_memory.reranker == "reranker_v1"
    assert updated.short_term_memory.session_key == "session_123"
    assert updated.short_term_memory.message_capacity == 12345


def find_config_file(filename: str, start_path: Path | None = None) -> Path:
    """Search parent directories for `sample_configs/filename`."""
    if start_path is None:
        start_path = Path(__file__).resolve()

    current = start_path.parent

    while current != current.parent:  # until we reach root
        candidate = current / "sample_configs" / filename
        if candidate.is_file():
            return candidate
        current = current.parent

    raise FileNotFoundError(
        f"Could not find '{filename}' in any parent 'sample_configs' folder.",
    )


def test_load_sample_cpu_config():
    config_path = find_config_file("episodic_memory_config.cpu.sample")
    conf = load_config_yml_file(str(config_path))
    resources_conf = conf.resources
    assert conf.logging.level == LogLevel.INFO
    assert conf.session_manager.database == "profile_storage"
    assert (
        len(resources_conf.language_models.openai_chat_completions_language_model_confs)
        > 0
    )
    assert (
        resources_conf.language_models.openai_chat_completions_language_model_confs[
            "ollama_model"
        ].model
        == "llama3"
    )
    postgres_conf = resources_conf.databases.relational_db_confs["profile_storage"]
    assert postgres_conf.password == SecretStr("<YOUR_PASSWORD_HERE>")
    assert conf.semantic_memory.database == "profile_storage"
    embedder_conf = resources_conf.embedders.openai["openai_embedder"]
    assert embedder_conf.api_key == SecretStr("<YOUR_API_KEY>")
    reranker_conf = resources_conf.rerankers.amazon_bedrock["aws_reranker_id"]
    assert reranker_conf.aws_access_key_id == SecretStr("<AWS_ACCESS_KEY_ID>")
    assert "Given" in conf.prompt.episode_summary_user_prompt
    assert "concise summary" in conf.prompt.episode_summary_system_prompt


def test_load_sample_gpu_config():
    config_path = find_config_file("episodic_memory_config.gpu.sample")
    conf = load_config_yml_file(str(config_path))
    assert conf.logging.level == LogLevel.INFO
