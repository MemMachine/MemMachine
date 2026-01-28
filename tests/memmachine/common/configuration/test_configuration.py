import logging
from pathlib import Path

import pytest
import yaml
from pydantic import SecretStr

from memmachine.common.configuration import (
    Configuration,
    EpisodicMemoryConfPartial,
)
from memmachine.common.configuration.episodic_config import (
    LongTermMemoryConfPartial,
    ShortTermMemoryConfPartial,
)
from memmachine.common.configuration.log_conf import LogLevel

logger = logging.getLogger(__name__)


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
    conf = Configuration.load_yml_file(str(config_path))
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
    assert isinstance(conf.prompt.episode_summary_user_prompt, str)
    assert len(conf.prompt.episode_summary_user_prompt) > 0
    assert "concise" in conf.prompt.episode_summary_system_prompt


def test_load_sample_gpu_config():
    config_path = find_config_file("episodic_memory_config.gpu.sample")
    conf = Configuration.load_yml_file(str(config_path))
    assert conf.logging.level == LogLevel.INFO


def test_serialize_and_deserialize_configuration():
    config_path = find_config_file("episodic_memory_config.gpu.sample")
    conf = Configuration.load_yml_file(str(config_path))
    yaml_str = conf.to_yaml()
    logger.debug("configuration yaml string:\n%s", yaml_str)
    data = yaml.safe_load(yaml_str)
    conf_cp = Configuration(**data)
    # Compare YAML output since _config_file_path differs (set vs None)
    assert conf_cp.to_yaml() == yaml_str


def test_load_yml_file_sets_config_file_path():
    """Test that load_yml_file stores the file path in the configuration."""
    config_path = find_config_file("episodic_memory_config.gpu.sample")
    conf = Configuration.load_yml_file(str(config_path))
    assert conf.config_file_path == str(config_path)


def test_config_file_path_none_when_not_loaded_from_file():
    """Test that config_file_path is None when configuration is created directly."""
    config_path = find_config_file("episodic_memory_config.gpu.sample")
    conf = Configuration.load_yml_file(str(config_path))
    # Create a new configuration from the same data (not from file)
    yaml_str = conf.to_yaml()
    data = yaml.safe_load(yaml_str)
    conf_direct = Configuration(**data)
    assert conf_direct.config_file_path is None


def test_save_configuration_to_file(tmp_path):
    """Test that save() writes configuration to a file."""
    config_path = find_config_file("episodic_memory_config.gpu.sample")
    conf = Configuration.load_yml_file(str(config_path))

    # Save to a new file
    save_path = tmp_path / "saved_config.yml"
    conf.save(str(save_path))

    # Verify file was created and can be loaded
    assert save_path.exists()
    loaded_conf = Configuration.load_yml_file(str(save_path))
    # Compare YAML output since _config_file_path differs between instances
    assert loaded_conf.to_yaml() == conf.to_yaml()


def test_save_configuration_to_original_path(tmp_path):
    """Test that save() writes to original path when no path is provided."""
    # First, create a config file in tmp_path
    config_path = find_config_file("episodic_memory_config.gpu.sample")
    original_conf = Configuration.load_yml_file(str(config_path))

    temp_config_path = tmp_path / "test_config.yml"
    temp_config_path.write_text(original_conf.to_yaml(), encoding="utf-8")

    # Load from temp path
    conf = Configuration.load_yml_file(str(temp_config_path))
    assert conf.config_file_path == str(temp_config_path)

    # Modify something (just to ensure save works)
    conf.server.port = 9999

    # Save without providing path (should use original path)
    conf.save()

    # Reload and verify
    reloaded = Configuration.load_yml_file(str(temp_config_path))
    assert reloaded.server.port == 9999


def test_save_raises_error_when_no_path():
    """Test that save() raises ValueError when no path is available."""
    config_path = find_config_file("episodic_memory_config.gpu.sample")
    conf = Configuration.load_yml_file(str(config_path))

    # Create a new configuration without file path
    yaml_str = conf.to_yaml()
    data = yaml.safe_load(yaml_str)
    conf_no_path = Configuration(**data)

    with pytest.raises(ValueError, match="No path provided"):
        conf_no_path.save()
