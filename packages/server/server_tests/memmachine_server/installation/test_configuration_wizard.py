from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from memmachine_server.common.configuration import Configuration, DatabasesConf
from memmachine_server.common.configuration.episodic_config import (
    EventLongTermMemoryConf,
)
from memmachine_server.common.episode_store import Episode
from memmachine_server.episodic_memory.short_term_memory.service_locator import (
    short_term_memory_params_from_config,
)
from memmachine_server.episodic_memory.short_term_memory.short_term_memory import (
    ShortTermMemory,
)
from memmachine_server.installation.configuration_wizard import (
    ConfigurationWizard,
)
from memmachine_server.main.memmachine import MemMachine


@pytest.fixture
def conf_args(tmp_path) -> ConfigurationWizard.Params:
    return ConfigurationWizard.Params(
        neo4j_provided=True,
        destination=str(tmp_path),
        prompt=False,
    )


@pytest.fixture
def generated_config(conf_args: ConfigurationWizard.Params) -> Configuration:
    with patch("builtins.input", return_value="api_key_value"):
        conf_file = ConfigurationWizard(conf_args).run_wizard()
    return Configuration.load_yml_file(conf_file)


@pytest.mark.parametrize(
    ("qdrant_available", "milvus_available", "vector_store"),
    [
        (True, True, ConfigurationWizard.QDRANT_VECTOR_STORE_ID),
        (False, True, ConfigurationWizard.MILVUS_VECTOR_STORE_ID),
        (False, False, ConfigurationWizard.SQLITE_VECTOR_STORE_ID),
    ],
)
def test_generated_config_resolves_default_stm_capacity(
    conf_args: ConfigurationWizard.Params,
    qdrant_available: bool,
    milvus_available: bool,
    vector_store: str,
) -> None:
    with (
        patch("builtins.input", return_value="api_key_value"),
        patch.object(
            ConfigurationWizard, "_qdrant_available", return_value=qdrant_available
        ),
        patch.object(
            ConfigurationWizard, "_milvus_available", return_value=milvus_available
        ),
    ):
        conf_file = ConfigurationWizard(conf_args).run_wizard()

    config = Configuration.load_yml_file(conf_file)
    memmachine = MemMachine(config, resources=Mock())
    resolved = memmachine._with_default_episodic_memory_conf(session_key="session1")

    assert isinstance(resolved.long_term_memory, EventLongTermMemoryConf)
    assert resolved.long_term_memory.vector_store == vector_store
    assert resolved.short_term_memory_enabled is True
    assert resolved.short_term_memory is not None
    assert resolved.short_term_memory.message_capacity == 64000


@pytest.mark.asyncio
@pytest.mark.parametrize("saved_summary", ["", "saved summary"])
async def test_generated_config_stm_summarizes_only_above_character_capacity(
    generated_config: Configuration, mock_llm_model: Mock, saved_summary: str
) -> None:
    memmachine = MemMachine(generated_config, resources=Mock())
    resolved = memmachine._with_default_episodic_memory_conf(session_key="session1")
    assert resolved.short_term_memory is not None
    assert resolved.short_term_memory.message_capacity == 64000

    mock_llm_model.generate_response.return_value = ("new summary", "")
    resources = Mock(
        get_session_data_manager=AsyncMock(return_value=None),
        get_language_model=AsyncMock(return_value=mock_llm_model),
    )
    params = await short_term_memory_params_from_config(
        resolved.short_term_memory, resources
    )
    memory = ShortTermMemory(params, summary=saved_summary)

    def episode(content: str) -> Episode:
        return Episode(
            uid=uuid4(),
            session_key="session1",
            content=content,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            producer_id="user1",
            producer_role="user",
        )

    try:
        # An ordinary turn exceeds the old 500-character installation budget.
        assert not await memory.add_episodes([episode("é" * 501)])
        assert await memory.get_summary() == saved_summary
        mock_llm_model.generate_response.assert_not_awaited()

        # Equality fits, including the summary. UTF-8 bytes would exceed it.
        remaining = 64000 - len(saved_summary) - 501
        assert not await memory.add_episodes([episode("x" * remaining)])
        assert await memory.get_summary() == saved_summary
        mock_llm_model.generate_response.assert_not_awaited()

        # One more Unicode code point triggers the real background consolidator.
        assert await memory.add_episodes([episode("🌍")])
        assert await memory.get_summary() == "new summary"
        mock_llm_model.generate_response.assert_awaited_once()
    finally:
        await memory.close()


@patch("builtins.input")
def test_configuration_wizard_all_default(mock_input, conf_args):
    mock_input.side_effect = ["api_key_value"]
    wizard = ConfigurationWizard(conf_args)
    conf_file = wizard.run_wizard()
    assert Path(conf_file).exists()
    config = Configuration.load_yml_file(conf_file)
    assert config is not None
    assert config.retrieval_agent.llm_model == ConfigurationWizard.LANGUAGE_MODEL_NAME
    assert config.retrieval_agent.reranker == ConfigurationWizard.RERANKER_NAME
    lm_confs = config.resources.language_models.openai_responses_language_model_confs
    assert len(lm_confs) == 1
    for lm_conf in lm_confs.values():
        assert lm_conf.api_key.get_secret_value() == "api_key_value"
    memmachine = MemMachine(config, resources=Mock())
    resolved = memmachine._with_default_episodic_memory_conf(session_key="session1")
    assert resolved.short_term_memory is not None
    assert resolved.short_term_memory.message_capacity == 64000


@patch.object(ConfigurationWizard, "_qdrant_available", return_value=False)
@patch.object(ConfigurationWizard, "_milvus_available", return_value=False)
@patch("builtins.input")
def test_configuration_with_prompt(mock_input, mock_milvus, mock_qdrant, conf_args):
    conf_args.prompt = True
    inputs = {
        # vector store prompt fires first; "" → accept default
        # (sqlite_vector_store when qdrant unavailable).
        "vector store": "",
        "language model provider": "openai",
        "openai llm model": "gpt-40-mini",
        "openai api key": "api_key_value",
        "openai base url": "https://api.my-openai.com/v1",
        "embedding model name": "text-embedding-3-small",
        "embedder dimensions": "1536",
        "api host for MemMachine": "127.0.0.1",
        "api port for MemMachine": "8088",
    }
    mock_input.side_effect = inputs.values()
    wizard = ConfigurationWizard(conf_args)
    conf_file = wizard.run_wizard()
    assert Path(conf_file).exists()
    config = Configuration.load_yml_file(conf_file)
    assert config is not None
    assert config.server.host == "127.0.0.1"
    assert config.server.port == 8088
    embedders = config.resources.embedders.openai
    assert len(embedders) == 1
    for embedder in embedders.values():
        assert embedder.model == "text-embedding-3-small"
        assert embedder.api_key.get_secret_value() == "api_key_value"
        assert embedder.base_url == "https://api.my-openai.com/v1"
        assert embedder.dimensions == 1536


@patch("builtins.input")
def test_configuration_neo4j(mock_input, conf_args):
    conf_args.neo4j_provided = False
    inputs = {
        "model api key": "api_key_value",
        "neo4j uri": "bolt://127.0.0.1:7687",
        "neo4j username": "neo4j_user",
        "neo4j password": "neo4j_password",
    }
    mock_input.side_effect = inputs.values()
    wizard = ConfigurationWizard(conf_args)
    conf_file = wizard.run_wizard()
    config = Configuration.load_yml_file(conf_file)
    neo4j_confs = config.resources.databases.neo4j_confs
    assert len(neo4j_confs) == 1
    for neo4j_conf in neo4j_confs.values():
        assert neo4j_conf.uri == "bolt://127.0.0.1:7687"
        assert neo4j_conf.user == "neo4j_user"
        assert neo4j_conf.password.get_secret_value() == "neo4j_password"


@patch.object(ConfigurationWizard, "_qdrant_available", return_value=False)
@patch.object(ConfigurationWizard, "_milvus_available", return_value=False)
@patch("builtins.input")
def test_configuration_wizard_aws_provider(
    mock_input, mock_milvus, mock_qdrant, conf_args
):
    conf_args.prompt = True
    inputs = {
        "vector store": "",
        "language model provider": "bedrock",
        "aws region": "us-west-2",
        "aws access key id": "key",
        "aws secret access key": "secret",
        "aws session token (optional)": "",
        "bedrock llm model": "openai.gpt-oss-30b-1:0",
        "bedrock embedder model": "amazon.titan-embed-text-v3",
        "api host for MemMachine": "127.0.0.1",
        "api port for MemMachine": "8088",
    }

    mock_input.side_effect = inputs.values()
    wizard = ConfigurationWizard(conf_args)
    conf_file = wizard.run_wizard()
    config = Configuration.load_yml_file(conf_file)
    embedders = config.resources.embedders.amazon_bedrock
    assert len(embedders) == 1
    for embedder in embedders.values():
        assert embedder.region == "us-west-2"
        assert embedder.aws_access_key_id is not None
        assert embedder.aws_access_key_id.get_secret_value() == "key"
        assert embedder.aws_secret_access_key is not None
        assert embedder.aws_secret_access_key.get_secret_value() == "secret"
        assert embedder.aws_session_token is None
        assert embedder.model_id == "amazon.titan-embed-text-v3"
    models = config.resources.language_models.amazon_bedrock_language_model_confs
    assert len(models) == 1
    for model in models.values():
        assert model.region == "us-west-2"
        assert model.aws_access_key_id is not None
        assert model.aws_access_key_id.get_secret_value() == "key"
        assert model.aws_secret_access_key is not None
        assert model.aws_secret_access_key.get_secret_value() == "secret"
        assert model.aws_session_token is None
        assert model.model_id == "openai.gpt-oss-30b-1:0"


@patch.object(ConfigurationWizard, "_qdrant_available", return_value=False)
@patch.object(ConfigurationWizard, "_milvus_available", return_value=False)
@patch("builtins.input")
def test_configuration_wizard_ollama_provider(
    mock_input, mock_milvus, mock_qdrant, conf_args
):
    conf_args.prompt = True
    inputs = {
        "vector store": "",
        "language model provider": "ollama",
        "ollama llm model": "llama4",
        "model api key": "key",
        "ollama base url": "http://localhost:11111/v1",
        "ollama embedder model": "nomic-embed-text-v2",
        "embedder dimensions": "1536",
        "api host for MemMachine": "127.0.0.1",
        "api port for MemMachine": "8088",
    }

    mock_input.side_effect = inputs.values()
    wizard = ConfigurationWizard(conf_args)
    conf_file = wizard.run_wizard()
    config = Configuration.load_yml_file(conf_file)
    embedders = config.resources.embedders.openai
    assert len(embedders) == 1
    for embedder in embedders.values():
        assert embedder.model == "nomic-embed-text-v2"
        assert embedder.api_key.get_secret_value() == "key"
        assert embedder.base_url == "http://localhost:11111/v1"
        assert embedder.dimensions == 1536
    models = (
        config.resources.language_models.openai_chat_completions_language_model_confs
    )
    assert len(models) == 1
    for model in models.values():
        assert model.model == "llama4"
        assert model.api_key.get_secret_value() == "key"
        assert model.base_url == "http://localhost:11111/v1"


def test_get_provided_database_config(conf_args):
    wizard = ConfigurationWizard(conf_args)
    db_conf = wizard.database_conf
    assert isinstance(db_conf, DatabasesConf)
    assert len(db_conf.neo4j_confs) == 1
    assert len(db_conf.relational_db_confs) == 1


@patch.object(ConfigurationWizard, "_qdrant_available", return_value=False)
@patch.object(ConfigurationWizard, "_milvus_available", return_value=False)
def test_vector_store_default_without_optional_backends_is_sqlite_compat(
    mock_milvus, mock_qdrant, conf_args
):
    """Without Qdrant the compatibility default is the USearch-backed SQLite
    vector store, NOT sqlite-vec — sqlite-vec depends on host SQLite having
    loadable-extension support, so it's never silently selected."""
    wizard = ConfigurationWizard(conf_args)
    assert wizard.vector_store_id == ConfigurationWizard.SQLITE_VECTOR_STORE_ID
    db_conf = wizard.database_conf
    assert len(db_conf.sqlite_vector_store_confs) == 1
    assert len(db_conf.sqlite_vec_vector_store_confs) == 0
    assert len(db_conf.qdrant_confs) == 0


@patch.object(ConfigurationWizard, "_qdrant_available", return_value=False)
@patch.object(ConfigurationWizard, "_milvus_available", return_value=True)
def test_vector_store_defaults_to_milvus_when_available_without_qdrant(
    mock_milvus, mock_qdrant, conf_args
):
    wizard = ConfigurationWizard(conf_args)
    assert wizard.vector_store_id == ConfigurationWizard.MILVUS_VECTOR_STORE_ID
    db_conf = wizard.database_conf
    assert len(db_conf.milvus_confs) == 1
    assert len(db_conf.qdrant_confs) == 0
    assert len(db_conf.sqlite_vector_store_confs) == 0
    assert len(db_conf.sqlite_vec_vector_store_confs) == 0


@patch.object(ConfigurationWizard, "_qdrant_available", return_value=True)
def test_vector_store_defaults_to_qdrant_when_available_silent(mock_qdrant, conf_args):
    # prompt=False (silent mode) -> Qdrant is the default when available.
    wizard = ConfigurationWizard(conf_args)
    assert wizard.vector_store_id == ConfigurationWizard.QDRANT_VECTOR_STORE_ID
    db_conf = wizard.database_conf
    assert len(db_conf.qdrant_confs) == 1
    assert len(db_conf.milvus_confs) == 0
    assert len(db_conf.sqlite_vector_store_confs) == 0
    assert len(db_conf.sqlite_vec_vector_store_confs) == 0


@patch.object(ConfigurationWizard, "_qdrant_available", return_value=True)
@patch.object(ConfigurationWizard, "_milvus_available", return_value=True)
@patch("builtins.input")
def test_vector_store_prompt_accepts_qdrant_default(
    mock_input, mock_milvus, mock_qdrant, conf_args
):
    conf_args.prompt = True
    # Empty input → keep the default (qdrant).
    mock_input.side_effect = ["", "api_key_value"]
    wizard = ConfigurationWizard(conf_args)
    assert wizard.vector_store_id == ConfigurationWizard.QDRANT_VECTOR_STORE_ID


@patch.object(ConfigurationWizard, "_qdrant_available", return_value=True)
@patch.object(ConfigurationWizard, "_milvus_available", return_value=True)
@patch("builtins.input")
def test_vector_store_prompt_accepts_sqlite_vector_store_override(
    mock_input, mock_milvus, mock_qdrant, conf_args
):
    conf_args.prompt = True
    mock_input.side_effect = ["sqlite_vector_store", "api_key_value"]
    wizard = ConfigurationWizard(conf_args)
    assert wizard.vector_store_id == ConfigurationWizard.SQLITE_VECTOR_STORE_ID
    db_conf = wizard.database_conf
    assert len(db_conf.sqlite_vector_store_confs) == 1
    assert len(db_conf.qdrant_confs) == 0
    assert len(db_conf.milvus_confs) == 0
    assert len(db_conf.sqlite_vec_vector_store_confs) == 0


@patch.object(ConfigurationWizard, "_qdrant_available", return_value=True)
@patch.object(ConfigurationWizard, "_milvus_available", return_value=True)
@patch("builtins.input")
def test_vector_store_prompt_accepts_milvus_override(
    mock_input, mock_milvus, mock_qdrant, conf_args
):
    conf_args.prompt = True
    mock_input.side_effect = ["milvus", "api_key_value"]
    wizard = ConfigurationWizard(conf_args)
    assert wizard.vector_store_id == ConfigurationWizard.MILVUS_VECTOR_STORE_ID
    db_conf = wizard.database_conf
    assert len(db_conf.milvus_confs) == 1
    assert len(db_conf.qdrant_confs) == 0
    assert len(db_conf.sqlite_vector_store_confs) == 0
    assert len(db_conf.sqlite_vec_vector_store_confs) == 0


@patch.object(ConfigurationWizard, "_qdrant_available", return_value=True)
@patch.object(ConfigurationWizard, "_milvus_available", return_value=True)
@patch("builtins.input")
def test_vector_store_prompt_accepts_sqlite_vec_override(
    mock_input, mock_milvus, mock_qdrant, conf_args
):
    conf_args.prompt = True
    mock_input.side_effect = ["sqlite_vec", "api_key_value"]
    wizard = ConfigurationWizard(conf_args)
    assert wizard.vector_store_id == ConfigurationWizard.SQLITE_VEC_VECTOR_STORE_ID
    db_conf = wizard.database_conf
    assert len(db_conf.sqlite_vec_vector_store_confs) == 1
    assert len(db_conf.qdrant_confs) == 0
    assert len(db_conf.milvus_confs) == 0
    assert len(db_conf.sqlite_vector_store_confs) == 0


@patch.object(ConfigurationWizard, "_qdrant_available", return_value=True)
@patch.object(ConfigurationWizard, "_milvus_available", return_value=True)
@patch("builtins.input")
def test_vector_store_prompt_reasks_on_invalid_then_accepts(
    mock_input, mock_milvus, mock_qdrant, conf_args
):
    """Garbage input doesn't silently fall through; the user has to retype."""
    conf_args.prompt = True
    mock_input.side_effect = ["garbage", "qdrant", "api_key_value"]
    wizard = ConfigurationWizard(conf_args)
    assert wizard.vector_store_id == ConfigurationWizard.QDRANT_VECTOR_STORE_ID


@patch("builtins.input")
def test_language_model_config(mock_input, conf_args):
    mock_input.side_effect = ["api_key_value"]
    conf = ConfigurationWizard(conf_args)
    lm_conf = conf.language_model_config
    assert len(lm_conf.openai_responses_language_model_confs) == 1
    assert len(lm_conf.amazon_bedrock_language_model_confs) == 0
    assert len(lm_conf.openai_chat_completions_language_model_confs) == 0


@patch("builtins.input")
def test_embedder_config(mock_input, conf_args):
    mock_input.side_effect = ["api_key_value"]
    conf = ConfigurationWizard(conf_args)
    embedder_conf = conf.embedders_conf
    assert len(embedder_conf.openai) == 1
    assert len(embedder_conf.amazon_bedrock) == 0
    assert len(embedder_conf.sentence_transformer) == 0


def test_reranker_config(conf_args):
    conf = ConfigurationWizard(conf_args)
    rerankers_conf = conf.rerankers_conf
    assert len(rerankers_conf.bm25) == 1
    assert len(rerankers_conf.identity) == 1
    assert len(rerankers_conf.rrf_hybrid) == 1
    assert len(rerankers_conf.amazon_bedrock) == 0
    assert len(rerankers_conf.cross_encoder) == 0


@patch("builtins.input")
def test_un_provided_neo4j(mock_input, conf_args):
    mock_input.side_effect = ["bolt://localhost:7687", "neo4j", "password"]
    conf_args.neo4j_provided = False
    conf = ConfigurationWizard(conf_args)
    db_conf = conf.database_conf
    assert len(db_conf.neo4j_confs) == 1
    neo4j_conf = db_conf.neo4j_confs[ConfigurationWizard.NEO4J_DB_ID]
    assert neo4j_conf.uri == "bolt://localhost:7687"
    assert neo4j_conf.user == "neo4j"
    assert neo4j_conf.password.get_secret_value() == "password"
    assert len(db_conf.relational_db_confs) == 1
