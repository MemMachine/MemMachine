import logging
import os
from dataclasses import dataclass

import yaml

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_BEDROCK_MODEL = "openai.gpt-oss-20b-1:0"
DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_BEDROCK_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
DEFAULT_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_BASE_URL = "http://host.docker.internal:11434/v1"

DEFAULT_PG_HOST = "localhost"
DEFAULT_PG_PORT = "5432"
DEFAULT_PG_USER = "postgres"
DEFAULT_PG_PASSWORD = "postgres_password"
DEFAULT_PG_DB = "postgres"

DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_NEO4J_USERNAME = "neo4j"
DEFAULT_NEO4J_PASSWORD = "neo4j"


@dataclass
class ConfigurationWizardArgs:
    neo4j_provided: bool
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    config_sources: dict[str, str]
    destination: str
    run_script_type: str


class ConfigurationWizard:
    def __init__(self, args: ConfigurationWizardArgs, logger: logging.Logger):
        self.logger: logging.Logger = logger
        self.neo4j_provided: bool = args.neo4j_provided
        self.config_sources: dict[str, str] = args.config_sources
        self.destination: str = args.destination
        self.configuration_path: str = os.path.join(
            self.destination, "configuration.yml"
        )
        if not os.path.exists(self.destination):
            self.logger.info(f"Creating configuration directory {self.destination}...")
            os.makedirs(self.destination)
        self.run_script_type: str = args.run_script_type
        if self.run_script_type not in ["bash", "powershell"]:
            self.logger.fatal(
                f"Invalid run script type {self.run_script_type}. Must be 'bash' or 'powershell'."
            )
            raise ValueError(
                f"Invalid run script type {self.run_script_type}. Must be 'bash' or 'powershell'."
            )

        self.config_type: str = ""
        self.provider: str = ""
        self.config_source: str = ""
        self.llm_model: str = ""
        self.embedding_model: str = ""

        self.api_key: str = ""
        self.aws_access_key_id: str = ""
        self.aws_secret_access_key: str = ""
        self.aws_region: str = ""
        self.ollama_base_url: str = ""

        self.pg_host: str = ""
        self.pg_port: int = 0
        self.pg_user: str = ""
        self.pg_password: str = ""
        self.pg_db: str = ""

        self.neo4j_uri: str = args.neo4j_uri if self.neo4j_provided else ""
        self.neo4j_username: str = args.neo4j_username if self.neo4j_provided else ""
        self.neo4j_password: str = args.neo4j_password if self.neo4j_provided else ""

        self.memmachine_host: str = ""
        self.memmachine_port: int = 0

    def extract_port_from_neo4j_uri(self, uri: str) -> int:
        try:
            parts = uri.split(":")
            port_part = parts[-1]
            port_str = port_part.split("/")[0]
            port = int(port_str)
            return port
        except (IndexError, ValueError):
            self.logger.fatal("Invalid Neo4j URI format.")
            raise ValueError("Invalid Neo4j URI format.")

    def extract_host_from_neo4j_uri(self, uri: str) -> str:
        try:
            parts = uri.split(":")
            host_part = parts[1]
            host = host_part.replace("//", "")
            return host
        except IndexError:
            self.logger.fatal("Invalid Neo4j URI format.")
            raise ValueError("Invalid Neo4j URI format.")

    def ask_config_type(self):
        config_type = (
            input("Which configuration would you like to use? (CPU/GPU) [CPU]: ")
            .strip()
            .upper()
            or "CPU"
        )
        if config_type in ["CPU", "GPU"]:
            self.config_type = config_type
        else:
            self.logger.warning("Invalid input. Defaulting to CPU.")
            self.config_type = "CPU"
        self.logger.info(f"{self.config_type} configuration selected.")
        self.config_source = self.config_sources.get(self.config_type, "")

    def ask_provider(self):
        provider = (
            input(
                "Which provider would you like to use? (OpenAI/Bedrock/Ollama) [OpenAI]: "
            )
            .strip()
            .upper()
            or "OPENAI"
        )
        if provider in ["OPENAI", "BEDROCK", "OLLAMA"]:
            self.provider = provider
        else:
            self.logger.warning("Invalid input. Defaulting to OPENAI.")
            self.provider = "OPENAI"
        self.logger.info(f"{self.provider} provider selected.")

    def ask_llm_model(self):
        match self.provider:
            case "OPENAI":
                self.llm_model = (
                    input(
                        f"Which OpenAI LLM model would you like to use? [{DEFAULT_OPENAI_MODEL}]: "
                    ).strip()
                    or DEFAULT_OPENAI_MODEL
                )
                self.logger.info(f"OpenAI model set to {self.llm_model}.")
            case "BEDROCK":
                self.llm_model = (
                    input(
                        f"Which AWS Bedrock LLM model would you like to use? [{DEFAULT_BEDROCK_MODEL}]: "
                    ).strip()
                    or DEFAULT_BEDROCK_MODEL
                )
                self.logger.info(f"Bedrock model set to {self.llm_model}.")
            case "OLLAMA":
                self.llm_model = (
                    input(
                        f"Which Ollama LLM model would you like to use? [{DEFAULT_OLLAMA_MODEL}]: "
                    ).strip()
                    or DEFAULT_OLLAMA_MODEL
                )
                self.logger.info(f"Ollama model set to {self.llm_model}.")
            case _:
                self.logger.error("Unsupported provider selected.")
                raise ValueError("Unsupported provider selected.")

    def ask_embedding_model(self):
        match self.provider:
            case "OPENAI":
                self.embedding_model = (
                    input(
                        f"Which OpenAI embedding model would you like to use? [{DEFAULT_OPENAI_EMBEDDING_MODEL}]: "
                    ).strip()
                    or DEFAULT_OPENAI_EMBEDDING_MODEL
                )
                self.logger.info(
                    f"Selected OpenAI embedding model: {self.embedding_model}."
                )
            case "BEDROCK":
                self.embedding_model = (
                    input(
                        f"Which AWS Bedrock embedding model would you like to use? [{DEFAULT_BEDROCK_EMBEDDING_MODEL}]: "
                    ).strip()
                    or DEFAULT_BEDROCK_EMBEDDING_MODEL
                )
                self.logger.info(
                    f"Selected AWS Bedrock embedding model: {self.embedding_model}."
                )
            case "OLLAMA":
                self.embedding_model = (
                    input(
                        f"Which Ollama embedding model would you like to use? [{DEFAULT_OLLAMA_EMBEDDING_MODEL}]: "
                    ).strip()
                    or DEFAULT_OLLAMA_EMBEDDING_MODEL
                )
                self.logger.info(
                    f"Selected Ollama embedding model: {self.embedding_model}."
                )
            case _:
                self.logger.error("Unsupported provider selected.")
                raise ValueError("Unsupported provider selected.")

    def parse_port(self, port_str: str) -> int:
        try:
            port = int(port_str)
            if 0 < port < 65536:
                return port
            else:
                self.logger.fatal("Port out of range.")
                raise ValueError("Port out of range.")
        except ValueError:
            self.logger.fatal("Invalid port number.")
            raise ValueError("Invalid port number.")

    def ask_pg_and_neo4j_configs(self):
        self.pg_host = (
            input(f"Enter PostgreSQL host [{DEFAULT_PG_HOST}]: ").strip()
            or DEFAULT_PG_HOST
        )
        self.pg_port = self.parse_port(
            input(f"Enter PostgreSQL port [{DEFAULT_PG_PORT}]: ").strip()
            or DEFAULT_PG_PORT
        )
        self.pg_user = (
            input(f"Enter PostgreSQL user [{DEFAULT_PG_USER}]: ").strip()
            or DEFAULT_PG_USER
        )
        self.pg_password = (
            input(f"Enter PostgreSQL password [{DEFAULT_PG_PASSWORD}]: ").strip()
            or DEFAULT_PG_PASSWORD
        )
        self.pg_db = (
            input(f"Enter PostgreSQL database name [{DEFAULT_PG_DB}]: ").strip()
            or DEFAULT_PG_DB
        )

        if not self.neo4j_provided:
            self.neo4j_uri = (
                input(f"Enter Neo4j URI [{DEFAULT_NEO4J_URI}]: ").strip()
                or DEFAULT_NEO4J_URI
            )
            self.neo4j_username = (
                input(f"Enter Neo4j username [{DEFAULT_NEO4J_USERNAME}]: ").strip()
                or DEFAULT_NEO4J_USERNAME
            )
            self.neo4j_password = (
                input(f"Enter Neo4j password [{DEFAULT_NEO4J_PASSWORD}]: ").strip()
                or DEFAULT_NEO4J_PASSWORD
            )

    def ask_api_key(self):
        match self.provider:
            case "OPENAI":
                self.api_key = input("Enter your OpenAI API key: ").strip()
            case "BEDROCK":
                self.aws_access_key_id = input("Enter your AWS Access Key ID: ").strip()
                self.aws_secret_access_key = input(
                    "Enter your AWS Secret Access Key: "
                ).strip()
                self.aws_region = input("Enter your AWS Region: ").strip()
            case "OLLAMA":
                self.ollama_base_url = (
                    input(f"Ollama base URL [{DEFAULT_OLLAMA_BASE_URL}]: ").strip()
                    or DEFAULT_OLLAMA_BASE_URL
                )

    def ask_host_and_port(self):
        self.memmachine_host = (
            input("Enter MemMachine host [localhost]: ").strip() or "localhost"
        )
        self.memmachine_port = self.parse_port(
            input("Enter MemMachine port [8080]: ").strip() or "8080"
        )

    def create_dot_env_file(self) -> None:
        envs: dict[str, str] = {
            "POSTGRES_HOST": self.pg_host,
            "POSTGRES_PORT": str(self.pg_port),
            "POSTGRES_USER": self.pg_user,
            "POSTGRES_PASSWORD": self.pg_password,
            "POSTGRES_DB": self.pg_db,
            "NEO4J_HOST": self.extract_host_from_neo4j_uri(self.neo4j_uri),
            "NEO4J_PORT": str(self.extract_port_from_neo4j_uri(self.neo4j_uri)),
            "NEO4J_USER": self.neo4j_username,
            "NEO4J_PASSWORD": self.neo4j_password,
            "MEMORY_CONFIG": self.configuration_path,
            "MCP_BASE_URL": f"http://{self.memmachine_host}:{self.memmachine_port}",
            "GATEWAY_HOST": f"http://{self.memmachine_host}:{self.memmachine_port}",
            "FAST_MCP_LOG_LEVEL": "INFO",
            "OPENAI_API_KEY": self.api_key,
            "LOG_LEVEL": "INFO",
            "PORT": str(self.memmachine_port),
            "HOST": self.memmachine_host,
        }

        env_file_path = os.path.join(self.destination, ".env")
        if os.path.exists(env_file_path):
            choice = (
                input(
                    f".env file already exists at {env_file_path}. Overwrite? (y/n): "
                )
                .strip()
                .lower()
            )
            if choice != "y":
                self.logger.info("Skipping .env file creation.")
                return
            os.remove(env_file_path)

        with open(env_file_path, "w") as file:
            for key, value in envs.items():
                file.write(f"{key}={value}\n")
        self.logger.info(f".env file created at {env_file_path}.")

    def create_configuration(self):
        with open(self.config_source, "r") as file:
            config_content = file.read()
        try:
            config = yaml.safe_load(config_content)
        except yaml.YAMLError as e:
            self.logger.fatal(f"Error parsing YAML configuration: {e}")
            raise ValueError("Invalid YAML configuration.")

        selected_embedder = ""
        selected_llm_model = ""
        match self.provider:
            case "OPENAI":
                selected_embedder = "openai_embedder"
                selected_llm_model = "openai_model"
                config["Model"]["openai_model"]["model"] = self.llm_model
                config["Model"]["openai_model"]["api_key"] = self.api_key
                config["embedder"]["openai_embedder"]["config"]["model"] = (
                    self.embedding_model
                )
                config["embedder"]["openai_embedder"]["config"]["api_key"] = (
                    self.api_key
                )
                self.logger.info(
                    f"Configured for OpenAI provider with LLM model {self.llm_model} and embedding model {self.embedding_model}."
                )
            case "BEDROCK":
                selected_embedder = "aws_embedder_id"
                selected_llm_model = "aws_model"
                config["Model"]["aws_model"]["model_id"] = self.llm_model
                config["Model"]["aws_model"]["region"] = self.aws_region
                config["Model"]["aws_model"]["aws_access_key_id"] = (
                    self.aws_access_key_id
                )
                config["Model"]["aws_model"]["aws_secret_access_key"] = (
                    self.aws_secret_access_key
                )
                config["embedder"]["aws_embedder_id"]["config"]["model_id"] = (
                    self.embedding_model
                )
                config["embedder"]["aws_embedder_id"]["config"]["region"] = (
                    self.aws_region
                )
                config["embedder"]["aws_embedder_id"]["config"]["aws_access_key_id"] = (
                    self.aws_access_key_id
                )
                config["embedder"]["aws_embedder_id"]["config"][
                    "aws_secret_access_key"
                ] = self.aws_secret_access_key
                self.logger.info(
                    f"Configured for AWS Bedrock provider with LLM model {self.llm_model} and embedding model {self.embedding_model}."
                )
            case "OLLAMA":
                selected_embedder = "ollama_embedder"
                selected_llm_model = "ollama_model"
                config["Model"]["ollama_model"]["model"] = self.llm_model
                config["Model"]["ollama_model"]["base_url"] = self.ollama_base_url
                config["embedder"]["ollama_embedder"]["config"]["model"] = (
                    self.embedding_model
                )
                config["embedder"]["ollama_embedder"]["config"]["base_url"] = (
                    self.ollama_base_url
                )
                self.logger.info(
                    f"Configured for Ollama provider with LLM model {self.llm_model} and embedding model {self.embedding_model}."
                )
            case _:
                self.logger.error("Unsupported provider selected.")
                raise ValueError("Unsupported provider selected.")
        config["long_term_memory"]["embedder"] = selected_embedder
        config["profile_memory"]["embedding_model"] = selected_embedder
        config["profile_memory"]["llm_model"] = selected_llm_model
        config["sessionMemory"]["model_name"] = selected_llm_model

        config["storage"]["profile_storage"]["host"] = self.pg_host
        config["storage"]["profile_storage"]["port"] = self.pg_port
        config["storage"]["profile_storage"]["user"] = self.pg_user
        config["storage"]["profile_storage"]["password"] = self.pg_password
        config["storage"]["profile_storage"]["db_name"] = self.pg_db

        config["vector_graph_store"]["my_storage_id"]["config"]["uri"] = self.neo4j_uri
        config["vector_graph_store"]["my_storage_id"]["config"]["username"] = (
            self.neo4j_username
        )
        config["vector_graph_store"]["my_storage_id"]["config"]["password"] = (
            self.neo4j_password
        )

        with open(self.configuration_path, "w") as file:
            yaml.dump(config, file)
        self.logger.info(f"Configuration written to {self.configuration_path}.")

    def run_wizard(self):
        self.ask_config_type()
        self.ask_provider()
        self.ask_llm_model()
        self.ask_embedding_model()
        self.ask_pg_and_neo4j_configs()
        self.ask_api_key()
        self.ask_host_and_port()

        self.create_configuration()
        self.create_dot_env_file()
