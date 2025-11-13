from pathlib import Path

import pytest
from pydantic import SecretStr

from memmachine.common.configuration import (
    EpisodicMemoryConf,
    EpisodicMemoryConfPartial,
    LongTermMemoryConf,
    LongTermMemoryConfPartial,
    SessionMemoryConf,
    SessionMemoryConfPartial,
    load_config_yml_file,
)
from memmachine.common.configuration.log_conf import LogLevel


@pytest.fixture
def long_term_memory_conf() -> LongTermMemoryConf:
    return LongTermMemoryConf(
        embedder="embedder_v1",
        reranker="reranker_v1",
        vector_graph_store="store_v1",
        enabled=True,
    )


def test_update_long_term_memory_conf(long_term_memory_conf: LongTermMemoryConf):
    update = LongTermMemoryConfPartial(
        embedder="embedder_v2",
    )

    updated = long_term_memory_conf.update(update)
    assert updated.embedder == "embedder_v2"
    assert updated.reranker == "reranker_v1"
    assert updated.vector_graph_store == "store_v1"
    assert updated.enabled is True


@pytest.fixture
def session_memory_conf() -> SessionMemoryConf:
    return SessionMemoryConf(
        model_name="model_v1",
        max_message_length=12345,
        enabled=True,
    )


def test_update_session_memory_conf(session_memory_conf: SessionMemoryConf):
    update = SessionMemoryConfPartial(
        max_token_num=3000,
        max_message_length=54321,
    )

    updated = session_memory_conf.update(update)
    assert updated.model_name == "model_v1"
    assert updated.max_token_num == 3000
    assert updated.max_message_length == 54321
    assert updated.message_capacity == 500
    assert updated.enabled is True


def test_update_episodic_memory_conf(
    long_term_memory_conf: LongTermMemoryConf, session_memory_conf: SessionMemoryConf
):
    base = EpisodicMemoryConf(
        sessionmemory=session_memory_conf,
        long_term_memory=long_term_memory_conf,
    )
    update = EpisodicMemoryConfPartial(
        long_term_memory=LongTermMemoryConfPartial(embedder="embedder_v2")
    )

    updated = base.update(update)
    assert updated.long_term_memory.embedder == "embedder_v2"
    assert updated.long_term_memory.reranker == "reranker_v1"
    assert updated.sessionmemory.max_message_length == 12345
    assert updated.sessionmemory.message_capacity == 500


def find_config_file(filename: str, start_path: Path = None) -> Path:
    """
    Search parent directories for `sample_configs/filename`.
    """
    if start_path is None:
        start_path = Path(__file__).resolve()

    current = start_path.parent

    while current != current.parent:  # until we reach root
        candidate = current / "sample_configs" / filename
        if candidate.is_file():
            return candidate
        current = current.parent

    raise FileNotFoundError(
        f"Could not find '{filename}' in any parent 'sample_configs' folder."
    )


def test_load_sample_cpu_config():
    config_path = find_config_file("episodic_memory_config.cpu.sample")
    conf = load_config_yml_file(str(config_path))
    assert conf.logging.level == LogLevel.INFO
    assert conf.sessiondb.uri == "sqlitetest.db"
    assert conf.model.openai_compatible_confs['ollama_model'].model == "llama3"
    postgres_conf = conf.storage.postgres_confs['profile_storage']
    assert postgres_conf.password == SecretStr("<YOUR_PASSWORD_HERE>")
    assert conf.profile_memory.database == "profile_storage"
    embedder_conf = conf.embedder.openai['openai_embedder']
    assert embedder_conf.api_key == SecretStr("<YOUR_API_KEY>")
    reranker_conf = conf.reranker.amazon_bedrock['aws_reranker_id']
    assert reranker_conf.aws_access_key_id == SecretStr("<AWS_ACCESS_KEY_ID>")
    assert "Given" in conf.prompt.episode_summary_user_prompt
    assert "concise summary" in conf.prompt.episode_summary_system_prompt


def test_load_sample_gpu_config():
    config_path = find_config_file("episodic_memory_config.gpu.sample")
    conf = load_config_yml_file(str(config_path))
    assert conf.logging.level == LogLevel.INFO
