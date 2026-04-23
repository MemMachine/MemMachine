"""Configuration wizard for MemMachine."""

import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import cast

from pydantic import SecretStr

from memmachine_server.common.configuration import (
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
from memmachine_server.common.configuration.database_conf import (
    AgeConf,
    DatabasesConf,
    Neo4jConf,
    SqlAlchemyConf,
    SupportedDB,
)
from memmachine_server.common.configuration.embedder_conf import (
    AmazonBedrockEmbedderConf,
    OpenAIEmbedderConf,
)
from memmachine_server.common.configuration.episodic_config import (
    EpisodicMemoryConfPartial,
    LongTermMemoryConfPartial,
    ShortTermMemoryConfPartial,
)
from memmachine_server.common.configuration.language_model_conf import (
    AmazonBedrockLanguageModelConf,
    OpenAIChatCompletionsLanguageModelConf,
    OpenAIResponsesLanguageModelConf,
)
from memmachine_server.common.configuration.reranker_conf import (
    BM25RerankerConf,
    IdentityRerankerConf,
    RRFHybridRerankerConf,
)
from memmachine_server.common.configuration.retrieval_config import RetrievalAgentConf
from memmachine_server.installation.utilities import (
    AGE_DEFAULTS,
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
    AGE_DB_ID = "age_db"
    # Companion SqlAlchemyConf pointing at the same Postgres instance as
    # AGE_DB_ID. Lets semantic memory's pgvector store share a database with
    # AGE's graph catalog — the whole point of the AGE backend.
    AGE_POSTGRES_DB_ID = "age_postgres_db"
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
        # "neo4j" (default) or "age". Picks which graph backend the wizard
        # writes into the generated configuration. Default preserves existing
        # installer behavior; AGE is the Apache-2.0 alternative.
        graph_backend: str = "neo4j"

    def __init__(self, args: Params) -> None:
        """Initialize the configuration wizard with parameters."""
        self.destination: Path = Path(args.destination)
        self.configuration_path: Path = Path(self.destination, "cfg.yml")
        self.prompt: bool = args.prompt
        self.neo4j_provided: bool = args.neo4j_provided
        if args.graph_backend not in ("neo4j", "age"):
            raise ValueError(
                f"graph_backend must be 'neo4j' or 'age' (got {args.graph_backend!r})"
            )
        self.graph_backend: str = args.graph_backend

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
            retrieval_agent=self.retrieval_agent_conf,
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
        """Generate semantic memory configuration.

        In AGE mode, semantic memory rides on the companion SqlAlchemyConf
        that points at the same Postgres instance as AGE — the single-stack
        promise. In Neo4j mode, semantic memory is left unconfigured
        (``database=None``) so the ``SemanticMemoryConf`` validator
        auto-disables it; the Neo4j wizard flow never provisioned a
        pgvector-capable relational database for semantic use.
        """
        return SemanticMemoryConf(
            llm_model=self.LANGUAGE_MODEL_NAME,
            embedding_model=self.EMBEDDER_NAME,
            database=(
                self.AGE_POSTGRES_DB_ID if self.graph_backend == "age" else None
            ),
            config_database=self.SQLITE_DB_ID,
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
            vector_graph_store=self.graph_db_id,
        )

    @property
    def graph_db_id(self) -> str:
        """DB id the long-term-memory ``vector_graph_store`` resolves to."""
        return self.AGE_DB_ID if self.graph_backend == "age" else self.NEO4J_DB_ID

    @cached_property
    def retrieval_agent_conf(self) -> RetrievalAgentConf:
        """Generate retrieval-agent configuration."""
        return RetrievalAgentConf(
            llm_model=self.LANGUAGE_MODEL_NAME,
            reranker=self.RERANKER_NAME,
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
        db_provider = SupportedDB.from_provider("sqlite")
        sqlite_db_conf = cast(
            SqlAlchemyConf,
            db_provider.build_config({"path": "memmachine.db"}),
        )
        if self.graph_backend == "age":
            # Register the same Postgres instance twice: once as an AgeConf
            # for the graph store, once as a SqlAlchemyConf so semantic
            # memory's pgvector store can ride along. Both rely on the
            # Dockerfile in deployments/docker/postgres-age/ shipping both
            # the ``age`` and ``vector`` extensions.
            age_conf = self.age_configs
            age_postgres_conf = SqlAlchemyConf(
                dialect="postgresql",
                driver="asyncpg",
                host=age_conf.host,
                port=age_conf.port,
                user=age_conf.user,
                password=age_conf.password,
                db_name=age_conf.db_name,
            )
            return DatabasesConf(
                age_confs={self.AGE_DB_ID: age_conf},
                relational_db_confs={
                    self.SQLITE_DB_ID: sqlite_db_conf,
                    self.AGE_POSTGRES_DB_ID: age_postgres_conf,
                },
            )
        return DatabasesConf(
            neo4j_confs={self.NEO4J_DB_ID: self.neo4j_configs},
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
    def age_configs(self) -> AgeConf:
        # Always prompt for AGE connection details. Unlike Neo4j, there's no
        # installer flow that pre-stands-up a local AGE-enabled Postgres, so
        # ``AGE_DEFAULTS`` is the presented default rather than an assumed
        # truth.
        age_host = (
            input(f"Enter AGE Postgres host [{AGE_DEFAULTS.host}]: ").strip()
            or AGE_DEFAULTS.host
        )
        age_port = int(
            input(f"Enter AGE Postgres port [{AGE_DEFAULTS.port}]: ").strip()
            or AGE_DEFAULTS.port
        )
        age_user = (
            input(f"Enter AGE Postgres user [{AGE_DEFAULTS.user}]: ").strip()
            or AGE_DEFAULTS.user
        )
        age_password = (
            input(f"Enter AGE Postgres password [{AGE_DEFAULTS.password}]: ").strip()
            or AGE_DEFAULTS.password
        )
        age_db_name = (
            input(f"Enter AGE database name [{AGE_DEFAULTS.database}]: ").strip()
            or AGE_DEFAULTS.database
        )
        age_graph_name = (
            input(f"Enter AGE graph name [{AGE_DEFAULTS.graph_name}]: ").strip()
            or AGE_DEFAULTS.graph_name
        )
        return AgeConf(
            host=age_host,
            port=age_port,
            user=age_user,
            password=SecretStr(age_password),
            db_name=age_db_name,
            graph_name=age_graph_name,
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
