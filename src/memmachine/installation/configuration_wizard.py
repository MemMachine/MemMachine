"""Configuration wizard for MemMachine."""

import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from pydantic import SecretStr

from memmachine.common.configuration import (
    Configuration,
    EmbeddersConf,
    EpisodeStoreConf,
    LanguageModelsConf,
    LogConf,
    RerankersConf,
    ResourcesConf,
    SemanticMemoryConf,
    ServerConf,
    SessionManagerConf,
)
from memmachine.common.configuration.database_conf import (
    DatabasesConf,
    Neo4jConf,
    SupportedDB,
)
from memmachine.common.configuration.embedder_conf import (
    AmazonBedrockEmbedderConf,
    OpenAIEmbedderConf,
)
from memmachine.common.configuration.episodic_config import (
    EpisodicMemoryConfPartial,
    LongTermMemoryConfPartial,
    ShortTermMemoryConfPartial,
)
from memmachine.common.configuration.language_model_conf import (
    AmazonBedrockLanguageModelConf,
    OpenAIChatCompletionsLanguageModelConf,
    OpenAIResponsesLanguageModelConf,
)
from memmachine.common.configuration.reranker_conf import (
    BM25RerankerConf,
    IdentityRerankerConf,
    RRFHybridRerankerConf,
)
from memmachine.installation.utilities import (
    DEFAULT_BEDROCK_EMBEDDING_MODEL,
    DEFAULT_BEDROCK_MODEL,
    DEFAULT_NEO4J_PASSWORD,
    DEFAULT_NEO4J_URI,
    DEFAULT_NEO4J_USERNAME,
    DEFAULT_OLLAMA_EMBEDDING_DIMENSIONS,
    DEFAULT_OLLAMA_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    DEFAULT_OPENAI_MODEL,
    ModelProvider,
)

logger = logging.getLogger(__name__)


class ConfigurationWizard:
    """Interactive configuration wizard for MemMachine."""

    NEO4J_DB_ID = "neo4j_db"
    SQLITE_DB_ID = "sqlite_db"
    LANGUAGE_MODEL_NAME = "llm_model"
    EMBEDDER_NAME = "my_embedder"
    RERANKER_NAME = "my_reranker"

    @dataclass
    class Params:
        """Parameters for the configuration wizard."""

        neo4j_provided: bool
        destination: str
        prompt: bool = False

    def __init__(self, args: Params) -> None:
        """Initialize the configuration wizard with parameters."""
        self.destination: Path = Path(args.destination)
        self.configuration_path: Path = Path(self.destination, "cfg.yml")
        self.prompt: bool = args.prompt
        self.neo4j_provided: bool = args.neo4j_provided

    def run_wizard(self) -> str:
        """Run the configuration wizard and write the configuration file."""
        config = self.config
        logger.info("Writing configuration to %s...", self.configuration_path)
        if not self.destination.exists():
            logger.info("Creating configuration directory %s...", self.destination)
            self.destination.mkdir(parents=True, exist_ok=True)
        with self.configuration_path.open("w", encoding="utf-8") as f:
            yaml_str = config.to_yaml()
            f.write(yaml_str)
        return str(self.configuration_path)

    @property
    def config(self) -> Configuration:
        """Generate the MemMachine configuration based on user input."""
        return Configuration(
            episodic_memory=self.episodic_memory_conf,
            semantic_memory=self.semantic_manager_conf,
            logging=self.log_conf,
            resources=self.resource_conf,
            session_manager=self.session_manager_config,
            episode_store=self.episode_store_config,
            server=self.server_conf,
        )

    @cached_property
    def server_conf(self) -> ServerConf:
        """Generate server configuration."""
        return ServerConf(host=self.host, port=int(self.port))

    @cached_property
    def semantic_manager_conf(self) -> SemanticMemoryConf:
        """Generate semantic memory configuration."""
        return SemanticMemoryConf(
            llm_model=self.LANGUAGE_MODEL_NAME,
            embedding_model=self.EMBEDDER_NAME,
            database=self.NEO4J_DB_ID,
        )

    @cached_property
    def episodic_memory_conf(self) -> EpisodicMemoryConfPartial:
        """Generate episodic memory configuration."""
        return EpisodicMemoryConfPartial(
            long_term_memory=self.long_term_memory_conf,
            short_term_memory=self.short_term_memory_conf,
        )

    @cached_property
    def long_term_memory_conf(self) -> LongTermMemoryConfPartial:
        """Generate long-term memory configuration."""
        return LongTermMemoryConfPartial(
            embedder=self.EMBEDDER_NAME,
            reranker=self.RERANKER_NAME,
            vector_graph_store=self.NEO4J_DB_ID,
        )

    @cached_property
    def short_term_memory_conf(self) -> ShortTermMemoryConfPartial:
        """Generate short-term memory configuration."""
        return ShortTermMemoryConfPartial(
            llm_model=self.LANGUAGE_MODEL_NAME,
            message_capacity=500,
        )

    @cached_property
    def model_provider(self) -> ModelProvider:
        """Prompt user to select a language model provider."""
        raw = self.ask_for(
            "Which provider would you like to use? (OpenAI/Bedrock/Ollama)", "OpenAI"
        )
        provider = ModelProvider.parse(raw)
        logger.info("%s provider selected.", provider.value)
        return provider

    @cached_property
    def language_model_config(self) -> LanguageModelsConf:
        """Generate language model configuration."""
        ret = LanguageModelsConf()
        match self.model_provider:
            case ModelProvider.OPENAI:
                conf = OpenAIResponsesLanguageModelConf(
                    model=self.open_ai_model_name,
                    api_key=SecretStr(self.api_key),
                    base_url=self.openai_base_url,
                )
                ret.openai_responses_language_model_confs[self.LANGUAGE_MODEL_NAME] = (
                    conf
                )
            case ModelProvider.BEDROCK:
                conf = AmazonBedrockLanguageModelConf(
                    region=self.aws_bedrock_region,
                    aws_access_key_id=SecretStr(self.aws_bedrock_access_key_id),
                    aws_secret_access_key=SecretStr(self.aws_bedrock_secret_access_key),
                    aws_session_token=self.aws_bedrock_session_token,
                    model_id=self.bedrock_model_name,
                )
                ret.amazon_bedrock_language_model_confs[self.LANGUAGE_MODEL_NAME] = conf
            case ModelProvider.OLLAMA:
                conf = OpenAIChatCompletionsLanguageModelConf(
                    model=self.ollama_model_name,
                    api_key=SecretStr(self.api_key),
                    base_url=self.ollama_base_url,
                )
                ret.openai_chat_completions_language_model_confs[
                    self.LANGUAGE_MODEL_NAME
                ] = conf
        return ret

    @cached_property
    def open_ai_model_name(self) -> str:
        return self.ask_for(
            "Enter OpenAI LLM model",
            DEFAULT_OPENAI_MODEL,
        )

    @cached_property
    def bedrock_model_name(self) -> str:
        return self.ask_for("Enter Bedrock model", DEFAULT_BEDROCK_MODEL)

    @cached_property
    def ollama_model_name(self) -> str:
        return self.ask_for(
            "Enter Ollama LLM model",
            DEFAULT_OLLAMA_MODEL,
        )

    @cached_property
    def ollama_base_url(self) -> str:
        return self.ask_for(
            "Enter Ollama base URL",
            "http://localhost:11434/v1",
        )

    @cached_property
    def openai_embedding_model_name(self) -> str:
        return self.ask_for(
            "Enter OpenAI embedding model", DEFAULT_OPENAI_EMBEDDING_MODEL
        )

    @cached_property
    def bedrock_embedding_model_name(self) -> str:
        return self.ask_for(
            "Enter Bedrock embedding model", DEFAULT_BEDROCK_EMBEDDING_MODEL
        )

    @cached_property
    def openai_base_url(self) -> str:
        return self.ask_for("Enter OpenAI base URL", DEFAULT_OPENAI_BASE_URL)

    @cached_property
    def ollama_embedding_model_name(self) -> str:
        return self.ask_for(
            "Enter Ollama embedding model",
            DEFAULT_OLLAMA_EMBEDDING_MODEL,
        )

    def ask_for(self, q: str, default: str) -> str:
        if not self.prompt:
            return default
        return input(f"{q} [{default}]: ").strip() or default

    @cached_property
    def embedder_dimensions(self) -> int:
        default_dimension = str(DEFAULT_OLLAMA_EMBEDDING_DIMENSIONS)
        dimension = self.ask_for("Enter embedding dimensions", default_dimension)
        return int(dimension)

    @cached_property
    def embedders_conf(self) -> EmbeddersConf:
        ret = EmbeddersConf()
        match self.model_provider:
            case ModelProvider.OPENAI:
                conf = OpenAIEmbedderConf(
                    model=self.openai_embedding_model_name,
                    dimensions=self.embedder_dimensions,
                    api_key=SecretStr(self.api_key),
                    base_url=self.openai_base_url,
                )
                ret.openai[self.EMBEDDER_NAME] = conf
            case ModelProvider.BEDROCK:
                conf = AmazonBedrockEmbedderConf(
                    region=self.aws_bedrock_region,
                    aws_access_key_id=SecretStr(self.aws_bedrock_access_key_id),
                    aws_secret_access_key=SecretStr(self.aws_bedrock_secret_access_key),
                    aws_session_token=self.aws_bedrock_session_token,
                    model_id=self.bedrock_embedding_model_name,
                )
                ret.amazon_bedrock[self.EMBEDDER_NAME] = conf
            case ModelProvider.OLLAMA:
                conf = OpenAIEmbedderConf(
                    model=self.ollama_embedding_model_name,
                    dimensions=self.embedder_dimensions,
                    api_key=SecretStr(self.api_key),
                    base_url=self.ollama_base_url,
                )
                ret.openai[self.EMBEDDER_NAME] = conf
        return ret

    @cached_property
    def log_conf(self) -> LogConf:
        return LogConf()

    @cached_property
    def episode_store_config(self) -> EpisodeStoreConf:
        return EpisodeStoreConf(database=self.SQLITE_DB_ID)

    @cached_property
    def session_manager_config(self) -> SessionManagerConf:
        return SessionManagerConf(database=self.SQLITE_DB_ID)

    @cached_property
    def database_conf(self) -> DatabasesConf:
        neo4j_db_conf = self.neo4j_configs
        db_provider = SupportedDB.from_provider("sqlite")
        sqlite_db_conf = db_provider.build_config({"path": "memmachine.db"})
        return DatabasesConf(
            neo4j_confs={self.NEO4J_DB_ID: neo4j_db_conf},
            relational_db_confs={self.SQLITE_DB_ID: sqlite_db_conf},
        )

    @cached_property
    def neo4j_configs(self) -> Neo4jConf:
        neo4j_uri = DEFAULT_NEO4J_URI
        neo4j_username = DEFAULT_NEO4J_USERNAME
        neo4j_password = DEFAULT_NEO4J_PASSWORD
        if not self.neo4j_provided:
            neo4j_uri = input(f"Enter Neo4j URI [{neo4j_uri}]: ").strip() or neo4j_uri
            neo4j_username = (
                input(f"Enter Neo4j username [{neo4j_username}]: ").strip()
                or neo4j_username
            )
            neo4j_password = (
                input(f"Enter Neo4j password [{neo4j_password}]: ").strip()
                or neo4j_password
            )
        return Neo4jConf(
            uri=neo4j_uri,
            user=neo4j_username,
            password=SecretStr(neo4j_password),
        )

    @cached_property
    def api_key(self) -> str:
        return input("Enter your Language Model API key: ").strip()

    @cached_property
    def aws_bedrock_access_key_id(self) -> str:
        return input("Enter your AWS Access Key ID: ").strip()

    @cached_property
    def aws_bedrock_secret_access_key(self) -> str:
        return input("Enter your AWS Secret Access Key: ").strip()

    @cached_property
    def aws_bedrock_session_token(self) -> SecretStr | None:
        token = input(
            "Enter your AWS Session Token (leave blank if not applicable): "
        ).strip()
        if len(token) == 0:
            return None
        return SecretStr(token)

    @cached_property
    def aws_bedrock_region(self) -> str:
        return input("Enter your AWS Region: ").strip()

    @cached_property
    def rerankers_conf(self) -> RerankersConf:
        ret = RerankersConf()
        ret.bm25["bm_ranker_id"] = BM25RerankerConf()
        ret.identity["id_ranker_id"] = IdentityRerankerConf()
        ret.rrf_hybrid[self.RERANKER_NAME] = RRFHybridRerankerConf(
            reranker_ids=["bm_ranker_id", "id_ranker_id"]
        )
        return ret

    @cached_property
    def resource_conf(self) -> ResourcesConf:
        return ResourcesConf(
            language_models=self.language_model_config,
            embedders=self.embedders_conf,
            rerankers=self.rerankers_conf,
            databases=self.database_conf,
        )

    @cached_property
    def host(self) -> str:
        return self.ask_for("Enter your API host", "localhost")

    @cached_property
    def port(self) -> str:
        return self.ask_for("Enter your API port", "8080")
