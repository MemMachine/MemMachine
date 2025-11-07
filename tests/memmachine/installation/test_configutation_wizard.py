import copy
import logging
import os
import shutil
from unittest.mock import patch

import yaml

from memmachine.installation.configuration_wizard import (
    DEFAULT_NEO4J_PASSWORD,
    DEFAULT_NEO4J_URI,
    DEFAULT_NEO4J_USERNAME,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_PG_DB,
    DEFAULT_PG_HOST,
    DEFAULT_PG_PASSWORD,
    DEFAULT_PG_PORT,
    DEFAULT_PG_USER,
    ConfigurationWizard,
    ConfigurationWizardArgs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("ConfigurationWizardTest")

test_file_dir = os.path.dirname(os.path.abspath(__file__))
sample_config_path = os.path.normpath(
    os.path.join(
        test_file_dir,
        "../../../src/memmachine/sample_configs",
    )
)

CONFIG_SOURCES = {
    "CPU": os.path.join(sample_config_path, "episodic_memory_config.cpu.sample"),
    "GPU": os.path.join(sample_config_path, "episodic_memory_config.gpu.sample"),
}

MOCK_DESTINATION_PATH = "/root/.config/memmachine"

DEFAULT_CONFIG_MAP = {
    "config_source": "CPU",
    "provider": "OpenAI",
    "model": DEFAULT_OPENAI_MODEL,
    "embedder_model": DEFAULT_OPENAI_EMBEDDING_MODEL,
    "postgres_host": DEFAULT_PG_HOST,
    "postgres_port": DEFAULT_PG_PORT,
    "postgres_user": DEFAULT_PG_USER,
    "postgres_password": DEFAULT_PG_PASSWORD,
    "postgres_db": DEFAULT_PG_DB,
    "neo4j_uri": DEFAULT_NEO4J_URI,
    "neo4j_username": DEFAULT_NEO4J_USERNAME,
    "neo4j_password": DEFAULT_NEO4J_PASSWORD,
    "openai_api_key": "",
    "aws_access_key": "",
    "aws_secret_key": "",
    "aws_region": "us-east-1",
    "ollama_base_url": DEFAULT_OLLAMA_BASE_URL,
    "memmachine_host": "localhost",
    "memmachine_port": "8080",
}

DEFAULT_EXPECTED_ENVS = {
    "POSTGRES_HOST": DEFAULT_PG_HOST,
    "POSTGRES_PORT": DEFAULT_PG_PORT,
    "POSTGRES_USER": DEFAULT_PG_USER,
    "POSTGRES_PASSWORD": DEFAULT_PG_PASSWORD,
    "POSTGRES_DB": DEFAULT_PG_DB,
    "NEO4J_HOST": "localhost",
    "NEO4J_PORT": "7687",
    "NEO4J_USER": "neo4j",
    "NEO4J_PASSWORD": "neo4j",
    "MEMORY_CONFIG": os.path.join(MOCK_DESTINATION_PATH, "configuration.yml"),
    "MCP_BASE_URL": "http://localhost:8080",
    "GATEWAY_HOST": "http://localhost:8080",
    "FAST_MCP_LOG_LEVEL": "INFO",
    "OPENAI_API_KEY": "",
    "LOG_LEVEL": "INFO",
    "PORT": "8080",
    "HOST": "localhost",
}

DEFAULT_EXPECTED_CONFIG = {
    "config_source": "CPU",
    "provider": "OpenAI",
    "model": DEFAULT_OPENAI_MODEL,
    "embedder_model": DEFAULT_OPENAI_EMBEDDING_MODEL,
    "postgres_config": {
        "vendor_name": "postgres",
        "host": DEFAULT_PG_HOST,
        "port": int(DEFAULT_PG_PORT),
        "user": DEFAULT_PG_USER,
        "password": DEFAULT_PG_PASSWORD,
        "db_name": DEFAULT_PG_DB,
    },
    "neo4j_config": {
        "uri": DEFAULT_NEO4J_URI,
        "username": DEFAULT_NEO4J_USERNAME,
        "password": DEFAULT_NEO4J_PASSWORD,
    },
    "credentials": "",
}

DEFAULT_WIZARD_ARGS = ConfigurationWizardArgs(
    neo4j_provided=False,
    neo4j_uri="bolt://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="neo4j",
    config_sources=CONFIG_SOURCES,
    destination=MOCK_DESTINATION_PATH,
    run_script_type="bash",
)


def get_config_source(config: dict[str, any]) -> str:
    assert "reranker" in config.keys(), f"Missing 'reranker' key in config, {config}"
    if "ce_ranker_id" in config["reranker"]["my_reranker_id"]["config"]["reranker_ids"]:
        return "GPU"
    return "CPU"


def get_provider(config: dict[str, any]) -> str:
    long_term_embedder = config["long_term_memory"]["embedder"]
    profile_embedder = config["profile_memory"]["embedding_model"]
    profile_model = config["profile_memory"]["llm_model"]
    session_model = config["sessionMemory"]["model_name"]
    assert long_term_embedder == profile_embedder
    assert profile_model == session_model
    if long_term_embedder == "openai_embedder" and profile_model == "openai_model":
        return "OpenAI"
    if long_term_embedder == "ollama_embedder" and profile_model == "ollama_model":
        return "Ollama"
    if long_term_embedder == "aws_embedder_id" and profile_model == "aws_model":
        return "AWS"
    assert False, "Unknown provider combination"


def get_credentials(config: dict[str, any]) -> str | dict[str, str]:
    provider = get_provider(config)
    if provider == "OpenAI":
        model_key = config["Model"]["openai_model"]["api_key"]
        embedder_key = config["embedder"]["openai_embedder"]["config"]["api_key"]
        assert model_key == embedder_key
        return model_key
    if provider == "Ollama":
        base_url = config["Model"]["ollama_model"]["base_url"]
        embedder_url = config["embedder"]["ollama_embedder"]["config"]["base_url"]
        assert base_url == embedder_url
        return base_url
    if provider == "AWS":
        access_key = config["Model"]["aws_model"]["aws_access_key_id"]
        secret_key = config["Model"]["aws_model"]["aws_secret_access_key"]
        region = config["Model"]["aws_model"]["region"]
        embedder_access_key = config["embedder"]["aws_embedder_id"]["config"][
            "aws_access_key_id"
        ]
        embedder_secret_key = config["embedder"]["aws_embedder_id"]["config"][
            "aws_secret_access_key"
        ]
        embedder_region = config["embedder"]["aws_embedder_id"]["config"]["region"]
        assert access_key == embedder_access_key
        assert secret_key == embedder_secret_key
        assert region == embedder_region
        return {"access_key": access_key, "secret_key": secret_key, "region": region}
    assert False, "Unknown provider"


def get_postgres_config(config: dict[str, any]) -> dict[str, str]:
    return config["storage"]["profile_storage"]


def get_neo4j_config(config: dict[str, any]) -> dict[str, str]:
    return config["vector_graph_store"]["my_storage_id"]["config"]


def get_model(config: dict[str, any]) -> str:
    provider = get_provider(config)
    if provider == "OpenAI":
        return config["Model"]["openai_model"]["model"]
    if provider == "Ollama":
        return config["Model"]["ollama_model"]["model"]
    if provider == "AWS":
        return config["Model"]["aws_model"]["model_id"]
    assert False, "Unknown provider"


def get_embedder_model(config: dict[str, any]) -> str:
    provider = get_provider(config)
    if provider == "OpenAI":
        return config["embedder"]["openai_embedder"]["config"]["model"]
    if provider == "Ollama":
        return config["embedder"]["ollama_embedder"]["config"]["model"]
    if provider == "AWS":
        return config["embedder"]["aws_embedder_id"]["config"]["model_id"]
    assert False, "Unknown provider"


def validate_configuration_file(expected_config: dict[str, any]):
    config_path = os.path.join(MOCK_DESTINATION_PATH, "configuration.yml")
    assert os.path.exists(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    assert get_config_source(config) == expected_config["config_source"]
    assert get_provider(config) == expected_config["provider"]
    assert get_model(config) == expected_config["model"]
    assert get_embedder_model(config) == expected_config["embedder_model"]
    assert get_postgres_config(config) == expected_config["postgres_config"]
    assert get_neo4j_config(config) == expected_config["neo4j_config"]
    assert get_credentials(config) == expected_config["credentials"]


def validate_environment_file(expected_envs: dict[str, str]):
    env_path = os.path.join(MOCK_DESTINATION_PATH, ".env")
    assert os.path.exists(env_path)
    envs = {}
    with open(env_path, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                envs[key] = value.strip('"')
    assert envs == expected_envs


@patch("builtins.input")
def test_configuration_wizard_all_default(mock_input, fs):
    os.makedirs(MOCK_DESTINATION_PATH, exist_ok=True)
    for source in CONFIG_SOURCES.values():
        fs.add_real_file(source)

    config_map = DEFAULT_CONFIG_MAP.copy()
    del config_map["aws_access_key"]
    del config_map["aws_secret_key"]
    del config_map["aws_region"]
    del config_map["ollama_base_url"]
    mock_input.side_effect = [""] * len(config_map)
    args = DEFAULT_WIZARD_ARGS
    wizard = ConfigurationWizard(args, logger=logger)
    wizard.run_wizard()

    validate_configuration_file(DEFAULT_EXPECTED_CONFIG)
    validate_environment_file(DEFAULT_EXPECTED_ENVS)

    shutil.rmtree(MOCK_DESTINATION_PATH)
    os.makedirs(MOCK_DESTINATION_PATH, exist_ok=False)
    mock_input.reset_mock()
    mock_input.side_effect = config_map.values()
    wizard.run_wizard()

    validate_configuration_file(DEFAULT_EXPECTED_CONFIG)
    validate_environment_file(DEFAULT_EXPECTED_ENVS)


@patch("builtins.input")
def test_configuration_gpu(mock_input, fs):
    os.makedirs(MOCK_DESTINATION_PATH, exist_ok=True)
    for source in CONFIG_SOURCES.values():
        fs.add_real_file(source)

    config_map = DEFAULT_CONFIG_MAP.copy()
    config_map["config_source"] = "GPU"
    del config_map["aws_access_key"]
    del config_map["aws_secret_key"]
    del config_map["aws_region"]
    del config_map["ollama_base_url"]
    mock_input.side_effect = config_map.values()
    args = DEFAULT_WIZARD_ARGS
    wizard = ConfigurationWizard(args, logger=logger)
    wizard.run_wizard()

    expected_config = DEFAULT_EXPECTED_CONFIG.copy()
    expected_config["config_source"] = "GPU"

    validate_configuration_file(expected_config)
    validate_environment_file(DEFAULT_EXPECTED_ENVS)


@patch("builtins.input")
def test_configuration_postgresql(mock_input, fs):
    os.makedirs(MOCK_DESTINATION_PATH, exist_ok=True)
    for source in CONFIG_SOURCES.values():
        fs.add_real_file(source)

    config_map = DEFAULT_CONFIG_MAP.copy()
    del config_map["aws_access_key"]
    del config_map["aws_secret_key"]
    del config_map["aws_region"]
    del config_map["ollama_base_url"]
    config_map["postgres_host"] = "remote-postgres-host"
    config_map["postgres_port"] = "2345"
    config_map["postgres_user"] = "custom_user"
    config_map["postgres_password"] = "custom_password"
    config_map["postgres_db"] = "custom_db"
    mock_input.side_effect = config_map.values()
    args = DEFAULT_WIZARD_ARGS
    wizard = ConfigurationWizard(args, logger=logger)
    wizard.run_wizard()

    expected_config = DEFAULT_EXPECTED_CONFIG.copy()
    expected_config["postgres_config"] = {
        "vendor_name": "postgres",
        "host": "remote-postgres-host",
        "port": 2345,
        "user": "custom_user",
        "password": "custom_password",
        "db_name": "custom_db",
    }
    expected_envs = DEFAULT_EXPECTED_ENVS.copy()
    expected_envs["POSTGRES_HOST"] = "remote-postgres-host"
    expected_envs["POSTGRES_PORT"] = "2345"
    expected_envs["POSTGRES_USER"] = "custom_user"
    expected_envs["POSTGRES_PASSWORD"] = "custom_password"
    expected_envs["POSTGRES_DB"] = "custom_db"

    validate_configuration_file(expected_config)
    validate_environment_file(expected_envs)


@patch("builtins.input")
def test_configuration_neo4j(mock_input, fs):
    os.makedirs(MOCK_DESTINATION_PATH, exist_ok=True)
    for source in CONFIG_SOURCES.values():
        fs.add_real_file(source)

    config_map = DEFAULT_CONFIG_MAP.copy()
    del config_map["aws_access_key"]
    del config_map["aws_secret_key"]
    del config_map["aws_region"]
    del config_map["ollama_base_url"]
    config_map["neo4j_uri"] = "bolt://remote-neo4j-host:7688"
    config_map["neo4j_username"] = "custom_neo4j_user"
    config_map["neo4j_password"] = "custom_neo4j_password"
    mock_input.side_effect = config_map.values()
    args = DEFAULT_WIZARD_ARGS
    wizard = ConfigurationWizard(args, logger=logger)
    wizard.run_wizard()

    expected_config = DEFAULT_EXPECTED_CONFIG.copy()
    expected_config["neo4j_config"] = {
        "uri": "bolt://remote-neo4j-host:7688",
        "username": "custom_neo4j_user",
        "password": "custom_neo4j_password",
    }
    expected_envs = DEFAULT_EXPECTED_ENVS.copy()
    expected_envs["NEO4J_HOST"] = "remote-neo4j-host"
    expected_envs["NEO4J_PORT"] = "7688"
    expected_envs["NEO4J_USER"] = "custom_neo4j_user"
    expected_envs["NEO4J_PASSWORD"] = "custom_neo4j_password"

    validate_configuration_file(expected_config)
    validate_environment_file(expected_envs)


@patch("builtins.input")
def test_configuration_wizard_openai_provider(mock_input, fs):
    os.makedirs(MOCK_DESTINATION_PATH, exist_ok=True)
    for source in CONFIG_SOURCES.values():
        fs.add_real_file(source)

    config_map = DEFAULT_CONFIG_MAP.copy()
    config_map["openai_api_key"] = "test_openai_api_key"
    del config_map["aws_access_key"]
    del config_map["aws_secret_key"]
    del config_map["aws_region"]
    del config_map["ollama_base_url"]
    mock_input.side_effect = config_map.values()
    args = DEFAULT_WIZARD_ARGS
    wizard = ConfigurationWizard(args, logger=logger)
    wizard.run_wizard()

    expected_config = DEFAULT_EXPECTED_CONFIG.copy()
    expected_config["provider"] = "OpenAI"
    expected_config["credentials"] = "test_openai_api_key"
    expected_envs = DEFAULT_EXPECTED_ENVS.copy()
    expected_envs["OPENAI_API_KEY"] = "test_openai_api_key"

    validate_configuration_file(expected_config)
    validate_environment_file(expected_envs)


@patch("builtins.input")
def test_configuration_wizard_aws_provider(mock_input, fs):
    os.makedirs(MOCK_DESTINATION_PATH, exist_ok=True)
    for source in CONFIG_SOURCES.values():
        fs.add_real_file(source)

    config_map = DEFAULT_CONFIG_MAP.copy()
    config_map["provider"] = "BEDROCK"
    config_map["aws_access_key"] = "test_aws_access_key"
    config_map["aws_secret_key"] = "test_aws_secret_key"
    del config_map["openai_api_key"]
    del config_map["ollama_base_url"]
    mock_input.side_effect = config_map.values()
    args = DEFAULT_WIZARD_ARGS
    wizard = ConfigurationWizard(args, logger=logger)
    wizard.run_wizard()

    expected_config = DEFAULT_EXPECTED_CONFIG.copy()
    expected_config["provider"] = "AWS"
    expected_config["credentials"] = {
        "access_key": "test_aws_access_key",
        "secret_key": "test_aws_secret_key",
        "region": "us-east-1",
    }
    expected_envs = DEFAULT_EXPECTED_ENVS.copy()

    validate_configuration_file(expected_config)
    validate_environment_file(expected_envs)


@patch("builtins.input")
def test_configuration_wizard_ollama_provider(mock_input, fs):
    os.makedirs(MOCK_DESTINATION_PATH, exist_ok=True)
    for source in CONFIG_SOURCES.values():
        fs.add_real_file(source)

    config_map = DEFAULT_CONFIG_MAP.copy()
    config_map["provider"] = "Ollama"
    config_map["ollama_base_url"] = "http://test-ollama:11434"
    del config_map["openai_api_key"]
    del config_map["aws_access_key"]
    del config_map["aws_secret_key"]
    del config_map["aws_region"]
    mock_input.side_effect = config_map.values()
    args = DEFAULT_WIZARD_ARGS
    wizard = ConfigurationWizard(args, logger=logger)
    wizard.run_wizard()

    expected_config = DEFAULT_EXPECTED_CONFIG.copy()
    expected_config["provider"] = "Ollama"
    expected_config["credentials"] = "http://test-ollama:11434"
    expected_envs = DEFAULT_EXPECTED_ENVS.copy()

    validate_configuration_file(expected_config)
    validate_environment_file(expected_envs)


@patch("builtins.input")
def test_configuration_wizard_neo4j_provided(mock_input, fs):
    os.makedirs(MOCK_DESTINATION_PATH, exist_ok=True)
    for source in CONFIG_SOURCES.values():
        fs.add_real_file(source)

    config_map = DEFAULT_CONFIG_MAP.copy()
    del config_map["aws_access_key"]
    del config_map["aws_secret_key"]
    del config_map["aws_region"]
    del config_map["ollama_base_url"]
    del config_map["neo4j_uri"]
    del config_map["neo4j_username"]
    del config_map["neo4j_password"]
    mock_input.side_effect = config_map.values()
    args = copy.deepcopy(DEFAULT_WIZARD_ARGS)
    args.neo4j_provided = True
    args.neo4j_uri = "bolt://test-neo4j-host:7688"
    args.neo4j_username = "test-neo4j-user"
    args.neo4j_password = "test-neo4j-password"
    wizard = ConfigurationWizard(args, logger=logger)
    wizard.run_wizard()
    expected_config = DEFAULT_EXPECTED_CONFIG.copy()
    expected_config["neo4j_config"] = {
        "uri": "bolt://test-neo4j-host:7688",
        "username": "test-neo4j-user",
        "password": "test-neo4j-password",
    }
    expected_envs = DEFAULT_EXPECTED_ENVS.copy()
    expected_envs["NEO4J_HOST"] = "test-neo4j-host"
    expected_envs["NEO4J_PORT"] = "7688"
    expected_envs["NEO4J_USER"] = "test-neo4j-user"
    expected_envs["NEO4J_PASSWORD"] = "test-neo4j-password"

    validate_configuration_file(expected_config)
    validate_environment_file(expected_envs)
