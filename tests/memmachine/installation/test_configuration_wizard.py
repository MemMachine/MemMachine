from pathlib import Path
from unittest.mock import patch

import pytest

from memmachine.common.configuration import Configuration, DatabasesConf
from memmachine.installation.configuration_wizard import (
    ConfigurationWizard,
)


@pytest.fixture
def conf_args(tmp_path) -> ConfigurationWizard.Params:
    return ConfigurationWizard.Params(
        neo4j_provided=True,
        destination=str(tmp_path),
        prompt=False,
    )


@patch("builtins.input")
def test_configuration_wizard_all_default(mock_input, conf_args):
    mock_input.side_effect = ["api_key_value"]
    wizard = ConfigurationWizard(conf_args)
    conf_file = wizard.run_wizard()
    assert Path(conf_file).exists()
    config = Configuration.load_yml_file(conf_file)
    assert config is not None
    lm_confs = config.resources.language_models.openai_responses_language_model_confs
    assert len(lm_confs) == 1
    for lm_conf in lm_confs.values():
        assert lm_conf.api_key.get_secret_value() == "api_key_value"


@patch("builtins.input")
def test_configuration_with_prompt(mock_input, conf_args):
    conf_args.prompt = True
    inputs = {
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
    prompt = config.prompt
    assert "profile_prompt" in prompt.session
    assert "profile_prompt" in prompt.profile


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


@patch("builtins.input")
def test_configuration_wizard_aws_provider(mock_input, conf_args):
    conf_args.prompt = True
    inputs = {
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
        assert embedder.aws_access_key_id.get_secret_value() == "key"
        assert embedder.aws_secret_access_key.get_secret_value() == "secret"
        assert embedder.aws_session_token is None
        assert embedder.model_id == "amazon.titan-embed-text-v3"
    models = config.resources.language_models.amazon_bedrock_language_model_confs
    assert len(models) == 1
    for model in models.values():
        assert model.region == "us-west-2"
        assert model.aws_access_key_id.get_secret_value() == "key"
        assert model.aws_secret_access_key.get_secret_value() == "secret"
        assert model.aws_session_token is None
        assert model.model_id == "openai.gpt-oss-30b-1:0"


@patch("builtins.input")
def test_configuration_wizard_ollama_provider(mock_input, conf_args):
    conf_args.prompt = True
    inputs = {
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
