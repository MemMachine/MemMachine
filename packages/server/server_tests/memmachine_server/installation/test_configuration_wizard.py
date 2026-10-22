import logging
import socket
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from memmachine_server.common.configuration import Configuration, DatabasesConf
from memmachine_server.common.configuration.database_conf import QdrantConf
from memmachine_server.installation.configuration_wizard import (
    ConfigurationCancelledError,
    ConfigurationWizard,
)


def prompt_responses(answers: Mapping[str, str | list[str]]) -> Callable[[str], str]:
    responses = {
        question: iter([answer] if isinstance(answer, str) else answer)
        for question, answer in answers.items()
    }

    def respond(prompt: str) -> str:
        matches = [
            reply
            for question, reply in responses.items()
            if prompt.startswith(question)
        ]
        assert len(matches) == 1, f"Unexpected prompt: {prompt}"
        return next(matches[0])

    return respond


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def fail_network(*args, **kwargs):
        pytest.fail("The configuration wizard must not contact external services")

    monkeypatch.setattr(socket, "getaddrinfo", fail_network)
    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setattr(socket.socket, "connect", fail_network)


@pytest.fixture
def no_qdrant_client(monkeypatch):
    monkeypatch.setitem(sys.modules, "qdrant_client", None)


@pytest.fixture
def conf_args(tmp_path) -> ConfigurationWizard.Params:
    return ConfigurationWizard.Params(
        neo4j_provided=True,
        destination=str(tmp_path),
        prompt=False,
    )


@pytest.fixture
def qdrant_inputs():
    return {
        "Choose vector store (": "",
        "Which provider would you like to use?": "",
        "Enter OpenAI LLM model": "",
        "Enter your Language Model API key:": "model-api-secret",
        "Enter OpenAI base URL": "",
        "Enter OpenAI embedding model": "",
        "Enter embedding dimensions": "",
        "Enter Qdrant host": "",
        "Enter Qdrant REST port": "",
        "Use this already-running external Qdrant server?": "yes",
        "Enter your API host": "",
        "Enter your API port": "",
    }


@patch("builtins.input")
def test_configuration_wizard_all_default(mock_input, conf_args):
    mock_input.side_effect = prompt_responses(
        {"Enter your Language Model API key:": "api_key_value"}
    )
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


@patch.object(ConfigurationWizard, "_qdrant_available", return_value=False)
@patch.object(ConfigurationWizard, "_milvus_available", return_value=False)
@patch("builtins.input")
def test_configuration_with_prompt(mock_input, mock_milvus, mock_qdrant, conf_args):
    conf_args.prompt = True
    inputs = {
        # vector store prompt fires first; "" → accept default
        # (sqlite_vector_store when qdrant unavailable).
        "Choose vector store (": "",
        "Which provider would you like to use?": "openai",
        "Enter OpenAI LLM model": "gpt-40-mini",
        "Enter your Language Model API key:": "api_key_value",
        "Enter OpenAI base URL": "https://api.my-openai.com/v1",
        "Enter OpenAI embedding model": "text-embedding-3-small",
        "Enter embedding dimensions": "1536",
        "Enter your API host": "127.0.0.1",
        "Enter your API port": "8088",
    }
    mock_input.side_effect = prompt_responses(inputs)
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
        "Enter your Language Model API key:": "api_key_value",
        "Enter Neo4j URI": "bolt://127.0.0.1:7687",
        "Enter Neo4j username": "neo4j_user",
        "Enter Neo4j password": "neo4j_password",
    }
    mock_input.side_effect = prompt_responses(inputs)
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
        "Choose vector store (": "",
        "Which provider would you like to use?": "bedrock",
        "Enter your AWS Region:": "us-west-2",
        "Enter your AWS Access Key ID:": "key",
        "Enter your AWS Secret Access Key:": "secret",
        "Enter your AWS Session Token": "",
        "Enter Bedrock model": "openai.gpt-oss-30b-1:0",
        "Enter Bedrock embedding model": "amazon.titan-embed-text-v3",
        "Enter your API host": "127.0.0.1",
        "Enter your API port": "8088",
    }

    mock_input.side_effect = prompt_responses(inputs)
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
        "Choose vector store (": "",
        "Which provider would you like to use?": "ollama",
        "Enter Ollama LLM model": "llama4",
        "Enter your Language Model API key:": "key",
        "Enter Ollama base URL": "http://localhost:11111/v1",
        "Enter Ollama embedding model": "nomic-embed-text-v2",
        "Enter embedding dimensions": "1536",
        "Enter your API host": "127.0.0.1",
        "Enter your API port": "8088",
    }

    mock_input.side_effect = prompt_responses(inputs)
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
    mock_milvus, mock_qdrant, conf_args, no_qdrant_client
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
    mock_milvus, mock_qdrant, conf_args, no_qdrant_client
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
    mock_input, mock_milvus, mock_qdrant, conf_args, no_qdrant_client
):
    conf_args.prompt = True
    # Empty input → keep the default (qdrant).
    mock_input.side_effect = prompt_responses(
        {
            "Choose vector store (": "",
            "Enter Qdrant host": "",
            "Enter Qdrant REST port": "",
            "Use this already-running external Qdrant server?": "y",
        }
    )
    wizard = ConfigurationWizard(conf_args)
    assert wizard.vector_store_id == ConfigurationWizard.QDRANT_VECTOR_STORE_ID
    assert (
        wizard.database_conf.qdrant_confs[wizard.QDRANT_VECTOR_STORE_ID] == QdrantConf()
    )


@patch.object(ConfigurationWizard, "_qdrant_available", return_value=True)
@patch.object(ConfigurationWizard, "_milvus_available", return_value=True)
@patch("builtins.input")
def test_vector_store_prompt_accepts_sqlite_vector_store_override(
    mock_input, mock_milvus, mock_qdrant, conf_args, no_qdrant_client
):
    conf_args.prompt = True
    mock_input.side_effect = prompt_responses(
        {"Choose vector store (": "sqlite_vector_store"}
    )
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
    mock_input, mock_milvus, mock_qdrant, conf_args, no_qdrant_client
):
    conf_args.prompt = True
    mock_input.side_effect = prompt_responses({"Choose vector store (": "milvus"})
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
    mock_input, mock_milvus, mock_qdrant, conf_args, no_qdrant_client
):
    conf_args.prompt = True
    mock_input.side_effect = prompt_responses({"Choose vector store (": "sqlite_vec"})
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
    mock_input.side_effect = prompt_responses(
        {"Choose vector store (": ["garbage", "qdrant"]}
    )
    wizard = ConfigurationWizard(conf_args)
    assert wizard.vector_store_id == ConfigurationWizard.QDRANT_VECTOR_STORE_ID


@pytest.mark.parametrize(
    ("host_input", "port_input", "host", "port", "endpoint"),
    [
        ("", "", "localhost", 6333, "http://localhost:6333"),
        (
            " qdrant.example.test ",
            "16333",
            "qdrant.example.test",
            16333,
            "http://qdrant.example.test:16333",
        ),
        ("::1", "65535", "::1", 65535, "http://[::1]:65535"),
        ("127.0.0.1", "1", "127.0.0.1", 1, "http://127.0.0.1:1"),
    ],
)
@patch.object(ConfigurationWizard, "_qdrant_available", return_value=True)
@patch.object(ConfigurationWizard, "_milvus_available", return_value=False)
@patch("builtins.input")
def test_qdrant_endpoint_is_confirmed_and_serialized(
    mock_input,
    mock_milvus,
    mock_qdrant,
    conf_args,
    qdrant_inputs,
    caplog,
    no_qdrant_client,
    host_input,
    port_input,
    host,
    port,
    endpoint,
):
    conf_args.prompt = True
    qdrant_inputs["Enter Qdrant host"] = host_input
    qdrant_inputs["Enter Qdrant REST port"] = port_input
    mock_input.side_effect = prompt_responses(qdrant_inputs)
    caplog.set_level(logging.INFO)

    wizard = ConfigurationWizard(conf_args)
    conf_file = wizard.run_wizard()
    config = Configuration.load_yml_file(conf_file)
    qdrant_conf = config.resources.databases.qdrant_confs[wizard.QDRANT_VECTOR_STORE_ID]
    assert qdrant_conf.host == host
    assert qdrant_conf.port == port
    assert f"Qdrant REST endpoint: {endpoint}." in caplog.messages
    assert f"Qdrant settings will be saved to {conf_file}." in caplog.text
    assert "already-running external server" in caplog.text
    assert "may share storage" in caplog.text
    assert "model-api-secret" not in caplog.text

    calls = mock_input.call_count
    assert (
        wizard.qdrant_conf
        is wizard.database_conf.qdrant_confs[wizard.QDRANT_VECTOR_STORE_ID]
    )
    assert (
        wizard.config.resources.databases.qdrant_confs[wizard.QDRANT_VECTOR_STORE_ID]
        == qdrant_conf
    )
    assert mock_input.call_count == calls


@patch.object(ConfigurationWizard, "_qdrant_available", return_value=True)
@patch.object(ConfigurationWizard, "_milvus_available", return_value=False)
@patch("builtins.input")
def test_qdrant_noninteractive_summary_without_new_prompts(
    mock_input, mock_milvus, mock_qdrant, conf_args, caplog, no_qdrant_client
):
    mock_input.side_effect = prompt_responses(
        {"Enter your Language Model API key:": "model-api-secret"}
    )
    caplog.set_level(logging.INFO)

    wizard = ConfigurationWizard(conf_args)
    conf_file = wizard.run_wizard()
    config = Configuration.load_yml_file(conf_file)
    assert (
        config.resources.databases.qdrant_confs[wizard.QDRANT_VECTOR_STORE_ID]
        == QdrantConf()
    )
    mock_input.assert_called_once_with("Enter your Language Model API key: ")
    assert "Qdrant REST endpoint: http://localhost:6333." in caplog.messages
    assert f"Qdrant settings will be saved to {conf_file}." in caplog.text
    assert "does not start it or check the connection" in caplog.text
    assert "model-api-secret" not in caplog.text


@pytest.mark.parametrize(
    "invalid_port", ["not-a-port", "0", "-1", "65536", "12.5", "port-secret"]
)
@patch("builtins.input")
def test_qdrant_retries_invalid_rest_port_without_echoing_input(
    mock_input, conf_args, caplog, invalid_port
):
    conf_args.prompt = True
    mock_input.side_effect = prompt_responses(
        {
            "Enter Qdrant host": "",
            "Enter Qdrant REST port": [invalid_port, "16333"],
            "Use this already-running external Qdrant server?": "yes",
        }
    )
    caplog.set_level(logging.INFO)

    config = ConfigurationWizard(conf_args).qdrant_conf
    assert config.port == 16333
    assert "Enter a Qdrant REST port between 1 and 65535." in caplog.messages
    assert "port-secret" not in caplog.text
    assert mock_input.call_count == 4


@pytest.mark.parametrize(
    "invalid_host",
    [
        "https://user:host-secret@qdrant.example.test:6333/path?api_key=token-secret",
        "qdrant.example.test:6333",
        "qdrant.example.test/path",
        "host name",
        "[::1]",
        "host\x1b[31m",
    ],
)
@patch("builtins.input")
def test_qdrant_retries_invalid_host_without_echoing_input(
    mock_input, conf_args, caplog, invalid_host
):
    conf_args.prompt = True
    mock_input.side_effect = prompt_responses(
        {
            "Enter Qdrant host": [invalid_host, "qdrant.example.test"],
            "Enter Qdrant REST port": "",
            "Use this already-running external Qdrant server?": "yes",
        }
    )
    caplog.set_level(logging.INFO)

    config = ConfigurationWizard(conf_args).qdrant_conf
    assert config.host == "qdrant.example.test"
    assert "without a scheme, port, path, credentials" in caplog.text
    assert "host-secret" not in caplog.text
    assert "token-secret" not in caplog.text
    assert mock_input.call_count == 4


@pytest.mark.parametrize("existing_config", [False, True])
@pytest.mark.parametrize("answer", ["", "n", "NO"])
@patch.object(ConfigurationWizard, "_qdrant_available", return_value=True)
@patch.object(ConfigurationWizard, "_milvus_available", return_value=False)
@patch("builtins.input")
def test_declining_qdrant_endpoint_does_not_write_configuration(
    mock_input,
    mock_milvus,
    mock_qdrant,
    conf_args,
    qdrant_inputs,
    tmp_path,
    no_qdrant_client,
    existing_config,
    answer,
):
    destination = tmp_path / "configuration"
    conf_file = destination / "cfg.yml"
    original_content = "keep this existing configuration\n"
    if existing_config:
        destination.mkdir()
        conf_file.write_text(original_content, encoding="utf-8")
    conf_args.destination = str(destination)
    conf_args.prompt = True
    qdrant_inputs["Use this already-running external Qdrant server?"] = answer
    mock_input.side_effect = prompt_responses(qdrant_inputs)

    with pytest.raises(
        ConfigurationCancelledError, match="Configuration was not written"
    ):
        ConfigurationWizard(conf_args).run_wizard()

    if existing_config:
        assert conf_file.read_text(encoding="utf-8") == original_content
    else:
        assert not destination.exists()


@patch("builtins.input")
def test_qdrant_requires_an_explicit_confirmation(mock_input, conf_args, caplog):
    conf_args.prompt = True
    mock_input.side_effect = prompt_responses(
        {
            "Enter Qdrant host": "",
            "Enter Qdrant REST port": "",
            "Use this already-running external Qdrant server?": [
                "confirmation-secret",
                "YES",
            ],
        }
    )
    caplog.set_level(logging.INFO)

    assert ConfigurationWizard(conf_args).qdrant_conf == QdrantConf()
    assert "Please answer yes or no." in caplog.messages
    assert "confirmation-secret" not in caplog.text
    assert mock_input.call_count == 4


@patch("builtins.input")
def test_qdrant_preserves_transport_and_authentication_settings(
    mock_input, conf_args, caplog
):
    original = QdrantConf(
        host="secure-qdrant",
        port=443,
        https=True,
        grpc_port=7443,
        prefer_grpc=True,
        api_key=SecretStr("qdrant-api-secret"),
        registry_replication_factor=3,
        metrics_factory_id="existing-metrics",
    )
    conf_args.prompt = True
    mock_input.side_effect = prompt_responses(
        {
            "Enter Qdrant host": "other-qdrant",
            "Enter Qdrant REST port": "8443",
            "Use this already-running external Qdrant server?": "yes",
        }
    )
    caplog.set_level(logging.INFO)

    with patch(
        "memmachine_server.installation.configuration_wizard.QdrantConf",
        return_value=original,
    ):
        config = ConfigurationWizard(conf_args).qdrant_conf

    assert config.host == "other-qdrant"
    assert config.port == 8443
    assert config.model_dump(exclude={"host", "port"}) == original.model_dump(
        exclude={"host", "port"}
    )
    assert original.host == "secure-qdrant"
    assert original.port == 443
    assert "Qdrant REST endpoint: https://other-qdrant:8443." in caplog.messages
    assert "Qdrant will use gRPC at other-qdrant:7443." in caplog.messages
    assert "qdrant-api-secret" not in caplog.text


@patch("builtins.input")
def test_language_model_config(mock_input, conf_args):
    mock_input.side_effect = prompt_responses(
        {"Enter your Language Model API key:": "api_key_value"}
    )
    conf = ConfigurationWizard(conf_args)
    lm_conf = conf.language_model_config
    assert len(lm_conf.openai_responses_language_model_confs) == 1
    assert len(lm_conf.amazon_bedrock_language_model_confs) == 0
    assert len(lm_conf.openai_chat_completions_language_model_confs) == 0


@patch("builtins.input")
def test_embedder_config(mock_input, conf_args):
    mock_input.side_effect = prompt_responses(
        {"Enter your Language Model API key:": "api_key_value"}
    )
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
    mock_input.side_effect = prompt_responses(
        {
            "Enter Neo4j URI": "bolt://localhost:7687",
            "Enter Neo4j username": "neo4j",
            "Enter Neo4j password": "password",
        }
    )
    conf_args.neo4j_provided = False
    conf = ConfigurationWizard(conf_args)
    db_conf = conf.database_conf
    assert len(db_conf.neo4j_confs) == 1
    neo4j_conf = db_conf.neo4j_confs[ConfigurationWizard.NEO4J_DB_ID]
    assert neo4j_conf.uri == "bolt://localhost:7687"
    assert neo4j_conf.user == "neo4j"
    assert neo4j_conf.password.get_secret_value() == "password"
    assert len(db_conf.relational_db_confs) == 1
