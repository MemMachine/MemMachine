"""API documentation strings for MemMachine server v2."""

from __future__ import annotations

from typing import ClassVar


class SpecDoc:
    """Common descriptions for API fields."""

    ORG_ID = """
    The unique identifier of the organization.

    - Must not contain slashes (`/`).
    - Must contain only letters, numbers, underscores, hyphens, colon, and Unicode
      characters (e.g., Chinese/Japanese/Korean). No slashes or other symbols
      are allowed.

    This value determines the namespace the project belongs to.
    """

    ORG_ID_RETURN = """
    The unique identifier of the organization this project belongs to.

    Returned exactly as stored by the system.
    """

    PROJECT_ID = """
    The identifier of the project.

    - Must be unique within the organization.
    - Must not contain slashes (`/`).
    - Must contain only letters, numbers, underscores, hyphens, colon, and Unicode
      characters (e.g., Chinese/Japanese/Korean). No slashes or other symbols
      are allowed.

    This ID is used in API paths and resource locations.
    """

    PROJECT_ID_RETURN = """
    The identifier of the project.

    This value uniquely identifies the project within the organization.
    """

    PROJECT_DESCRIPTION = """
    A human-readable description of the project.
    Used for display purposes in UIs and dashboards.
    Optional; defaults to an empty string.
    """

    PROJECT_CONFIG = """
    Configuration settings associated with this project.

    Defines which models (reranker, embedder) to use. If any values within
    `ProjectConfig` are empty, global defaults are applied.
    """

    RERANKER_ID = """
    The name of the reranker model to use for this project.

    - Must refer to a reranker model defined in the system configuration.
    - If set to an empty string (default), the globally configured reranker will
      be used.

    Rerankers typically re-score retrieved documents to improve result quality.
    """

    EMBEDDER_ID = """
    The name of the embedder model to use for this project.

    - Must refer to an embedder model defined in the system configuration.
    - If set to an empty string (default), the globally configured embedder will
      be used.

    Embedders generate vector embeddings for text to support semantic search and
    similarity operations.
    """

    EPISODE_COUNT = "The total number of episodic memories in the project."

    EPISODE_CONTENT = "The content payload of the episode."

    EPISODE_PRODUCER_ID = "Identifier of the episode producer."

    EPISODE_PRODUCER_ROLE = "Role of the producer (e.g., user/assistant/system)."

    EPISODE_PRODUCED_FOR_ID = "Identifier of the intended recipient of the episode."

    EPISODE_TYPE = "The type of episode being stored (e.g., message)."

    EPISODE_METADATA = "Optional metadata associated with the episode."

    EPISODE_CREATED_AT = "Timestamp when the episode was created."

    EPISODE_UID = "Unique identifier for the episode."

    EPISODE_SCORE = "Optional relevance score for the episode."

    EPISODE_SESSION_KEY = "Session key associated with the episode."

    EPISODE_SEQUENCE_NUM = "Sequence number within the session."

    EPISODE_CONTENT_TYPE = "Content type of the episode."

    EPISODE_FILTERABLE_METADATA = "Metadata indexed for filtering."

    SEMANTIC_SET_ID = "Identifier of the semantic set."

    SEMANTIC_CATEGORY = "Category of the semantic feature."

    SEMANTIC_TAG = "Tag associated with the semantic feature."

    SEMANTIC_FEATURE_NAME = "Name of the semantic feature."

    SEMANTIC_VALUE = "Value of the semantic feature."

    SEMANTIC_METADATA = "Storage metadata for the semantic feature."

    SEMANTIC_METADATA_CITATIONS = "Episode IDs cited by this semantic feature."

    SEMANTIC_METADATA_ID = "Identifier for the semantic feature."

    SEMANTIC_METADATA_OTHER = "Additional storage metadata for the semantic feature."

    FEATURE_ID = "Unique identifier for the semantic feature."

    FEATURE_SET_ID = "Identifier of the semantic set to add the feature to."

    FEATURE_CATEGORY_NAME = "Category name to attach the feature to."

    FEATURE_TAG = "Tag name to associate with the feature."

    FEATURE_NAME = "Feature name/key."

    FEATURE_VALUE = "Feature value."

    FEATURE_METADATA = "Optional metadata to store alongside the feature."

    FEATURE_CITATIONS = "Optional episode IDs supporting this feature."

    FEATURE_LOAD_CITATIONS = "Whether to load referenced episode IDs."

    EPISODIC_SHORT_EPISODES = "Matched short-term episodic entries."

    EPISODIC_SHORT_SUMMARY = "Summaries of matched short-term episodes."

    EPISODIC_LONG_EPISODES = "Matched long-term episodic entries."

    EPISODIC_LONG_TERM = "Long-term episodic search results."

    EPISODIC_SHORT_TERM = "Short-term episodic search results."

    SEARCH_EPISODIC_MEMORY = "Episodic memory search results."

    SEARCH_SEMANTIC_MEMORY = "Semantic memory search results."

    LIST_EPISODIC_MEMORY = "Listed episodic memory entries."

    LIST_SEMANTIC_MEMORY = "Listed semantic memory entries."

    MEMORY_CONTENT = "The content or text of the message."

    MEMORY_PRODUCER = """
    The sender of the message. This is a user-friendly name for
    the LLM to understand the message context. Defaults to 'user'.
    """

    MEMORY_PRODUCE_FOR = """
    The intended recipient of the message. This is a user-friendly name for
    the LLM to understand the message context. Defaults to an empty string.
    """

    MEMORY_TIMESTAMP = """
    The timestamp when the message was created, in ISO 8601 format.
    The formats supported are:
    - ISO 8601 string (e.g., '2023-10-01T12:00:00Z' or '2023-10-01T08:00:00-04:00')
    - Unix epoch time in seconds (e.g., 1633072800)
    - Unix epoch time in milliseconds (e.g., 1633072800000)
    If not provided, the server assigns the current time.
    If the format is unrecognized, an error is returned.
    """

    MEMORY_ROLE = """
    The role of the message in a conversation (e.g., 'user', 'assistant',
    'system'). Optional; defaults to an empty string.
    """

    MEMORY_METADATA = """
    Additional metadata associated with the message, represented as key-value
    pairs. Optional; defaults to an empty object.
    Retrieval operations may utilize this metadata for filtering.
    Use 'metadata.{key}' to filter based on specific metadata keys.
    """

    MEMORY_EPISODIC_TYPE = """
    The type of an episode (e.g., 'message').
    """

    MEMORY_MESSAGES = """
    A list of messages to be added (batch input).
    Must contain at least one message.
    """

    MEMORY_UID = "The unique identifier of the memory message."

    ADD_MEMORY_RESULTS = "The list of results for each added memory message."

    TOP_K = """
    The maximum number of memories to return in the search results.
    """

    EXPAND_CONTEXT = """
    The number of additional episodes to include around each matched
    episode from long term memory for better context.
    """

    SCORE_THRESHOLD = """
    The minimum score for a memory to be included in the search results. Defaults
    to -inf (no threshold) represented as None. Meaningful only for certain ranking methods.
    """

    QUERY = """
    The natural language query used for semantic memory search. This should be
    a descriptive string of the information you are looking for.
    """

    FILTER_MEM = """
    An optional string filter applied to the memory metadata. This uses a
    simple query language (e.g., 'metadata.user_id=123') for exact matches.
    Multiple conditions can be combined using AND operators.  The metadata
    fields are prefixed with 'metadata.' to distinguish them from other fields.
    """

    MEMORY_TYPES = """
    A list of memory types to include in the search (e.g., episodic, semantic).
    If empty, all available types are searched.
    """

    PAGE_SIZE = """
    The maximum number of memories to return per page. Use this for pagination.
    """

    PAGE_NUM = """
    The zero-based page number to retrieve. Use this for pagination.
    """

    MEMORY_TYPE_SINGLE = """
    The specific memory type to list (e.g., episodic or semantic).
    """

    EPISODIC_ID = """
    The unique ID of the specific episodic memory.
    """

    EPISODIC_IDS = """
    A list of unique IDs of episodic memories."""

    SEMANTIC_ID = """
    The unique ID of the specific semantic memory.
    """

    SEMANTIC_IDS = """
    A list of unique IDs of semantic memories."""

    STATUS = """
    The status code of the search operation. 0 typically indicates success.
    """

    CONTENT = """
    The dictionary containing the memory search results (e.g., list of memory
    objects).
    """

    ERROR_CODE = """
    The http status code if the operation failed."""

    ERROR_MESSAGE = """
    A descriptive error message if the operation failed."""

    ERROR_EXCEPTION = """
    The exception details if an error occurred during the operation."""

    ERROR_TRACE = """
    The stack trace of the exception if available."""

    ERROR_INTERNAL = """
    The real error that triggered the exception, for internal debugging."""

    SERVER_VERSION = """
    The version of the MemMachine server."""

    CLIENT_VERSION = """
    The version of the MemMachine client."""

    RESOURCE_NAME = """
    The unique name/identifier of the resource."""

    RESOURCE_PROVIDER = """
    The provider type of the resource (e.g., 'openai', 'amazon-bedrock"""

    RESOURCE_STATUS = """
    The current status of the resource."""

    RESOURCES_STATUS = """
    The status of all configured resources."""

    RESOURCE_ERROR_MSG = """
    The error message if the resource operation failed."""

    EMBEDDERS_STATUS = """
    The status of all configured embedders."""

    LANGUAGE_MODELS_STATUS = """
    The status of all configured language models."""

    RERANKERS_STATUS = """
    The status of all configured rerankers."""

    DATABASES_STATUS = """
    The status of all configured databases."""

    AGENT_MODE = """
    Whether to use retrieval agent for memory search or not. Agent Mode does not
    support score threshold filtering."""

    # --- Configuration API Fields ---

    API_KEY_OPENAI = """
    OpenAI API key for authentication."""

    MODEL_NAME = """
    The model name to use."""

    EMBEDDING_DIMENSIONS = """
    The number of dimensions for embedding vectors."""

    API_BASE_URL = """
    Custom base URL for the API endpoint."""

    MAX_INPUT_LENGTH = """
    Maximum input length in Unicode code points."""

    MAX_RETRY_INTERVAL = """
    Maximum retry interval in seconds for failed requests."""

    AWS_REGION = """
    AWS region where the service is hosted."""

    MODEL_ID_BEDROCK = """
    Amazon Bedrock model ID."""

    AWS_ACCESS_KEY_ID = """
    AWS access key ID for authentication."""

    AWS_SECRET_ACCESS_KEY = """
    AWS secret access key for authentication."""

    AWS_SESSION_TOKEN = """
    AWS session token for temporary credentials."""

    SENTENCE_TRANSFORMER_MODEL = """
    The name of the sentence transformer model to use."""

    EMBEDDER_NAME = """
    Unique name/identifier for the embedder."""

    EMBEDDER_PROVIDER_TYPE = """
    The embedder provider type (e.g., 'openai', 'amazon-bedrock', 'sentence-transformer')."""

    PROVIDER_CONFIG = """
    Provider-specific configuration settings."""

    LM_NAME = """
    Unique name/identifier for the language model."""

    LM_PROVIDER_TYPE = """
    The language model provider type (e.g., 'openai-responses', 'openai-chat-completions', 'amazon-bedrock')."""

    INFERENCE_CONFIG = """
    Inference configuration for the model."""

    ADDITIONAL_MODEL_FIELDS = """
    Additional model request fields."""

    OPERATION_SUCCESS = """
    Whether the operation succeeded."""

    RESOURCE_STATUS_AFTER_OP = """
    Current status of the resource after the operation."""

    OPERATION_ERROR = """
    Error message if the operation failed."""

    DELETION_SUCCESS = """
    Whether the deletion succeeded."""

    STATUS_MESSAGE = """
    Status message describing the result of the operation."""

    # --- Memory Configuration Update Fields ---

    LTM_EMBEDDER = """
    The ID of the embedder resource to use for long-term memory.
    Must reference an embedder configured in the resources section."""

    LTM_RERANKER = """
    The ID of the reranker resource to use for long-term memory search.
    Must reference a reranker configured in the resources section."""

    LTM_VECTOR_GRAPH_STORE = """
    The ID of the vector graph store (database) for storing long-term memories.
    Must reference a database configured in the resources section."""

    STM_LLM_MODEL = """
    The ID of the language model to use for short-term memory summarization.
    Must reference a language model configured in the resources section."""

    STM_MESSAGE_CAPACITY = """
    The maximum message capacity for short-term memory, in characters.
    When exceeded, older messages are summarized."""

    EPISODIC_LTM_UPDATE = """
    Partial update for long-term memory settings. Only supplied fields
    are updated; omitted fields remain unchanged."""

    EPISODIC_STM_UPDATE = """
    Partial update for short-term memory settings. Only supplied fields
    are updated; omitted fields remain unchanged."""

    EPISODIC_LTM_ENABLED = """
    Whether long-term episodic memory is enabled."""

    EPISODIC_STM_ENABLED = """
    Whether short-term episodic memory is enabled."""

    EPISODIC_ENABLED = """
    Whether episodic memory as a whole is enabled."""

    SEMANTIC_ENABLED = """
    Whether semantic memory is enabled. When set to true, the required
    fields (database, llm_model, embedding_model) must also be configured.
    Set to false to disable semantic memory entirely."""

    SEMANTIC_DATABASE = """
    The ID of the database to use for semantic memory storage.
    Must reference a database configured in the resources section."""

    SEMANTIC_LLM_MODEL = """
    The ID of the language model to use for semantic memory extraction.
    Must reference a language model configured in the resources section."""

    SEMANTIC_EMBEDDING_MODEL = """
    The ID of the embedder to use for semantic memory vector search.
    Must reference an embedder configured in the resources section."""

    SEMANTIC_INGESTION_MESSAGES = """
    The number of uningested messages that triggers an ingestion cycle."""

    SEMANTIC_INGESTION_AGE = """
    The maximum age (in seconds) of uningested messages before
    triggering an ingestion cycle."""

    UPDATE_EPISODIC_MEMORY = """
    Partial update for episodic memory configuration. Only supplied
    fields are updated; omitted fields remain unchanged."""

    UPDATE_SEMANTIC_MEMORY = """
    Partial update for semantic memory configuration. Only supplied
    fields are updated; omitted fields remain unchanged."""

    # --- Memory Configuration Response Fields ---

    LTM_CONFIG = """
    Current long-term memory configuration."""

    STM_CONFIG = """
    Current short-term memory configuration."""

    LTM_CONFIG_EMBEDDER = """
    ID of the embedder resource used for long-term memory."""

    LTM_CONFIG_RERANKER = """
    ID of the reranker resource used for long-term memory search."""

    LTM_CONFIG_VECTOR_GRAPH_STORE = """
    ID of the vector graph store (database) for storing long-term memories."""

    LTM_CONFIG_ENABLED = """
    Whether long-term memory is enabled."""

    STM_CONFIG_LLM_MODEL = """
    ID of the language model used for short-term memory summarization."""

    STM_CONFIG_MESSAGE_CAPACITY = """
    Maximum message capacity for short-term memory, in characters."""

    STM_CONFIG_ENABLED = """
    Whether short-term memory is enabled."""

    EPISODIC_MEMORY_CONFIG = """
    Current episodic memory configuration including both long-term and short-term memory."""

    SEMANTIC_MEMORY_CONFIG = """
    Current semantic memory configuration."""

    # --- Semantic Set Type Fields ---

    SET_TYPE_ID = """
    Unique identifier for the semantic set type."""

    SET_TYPE_IS_ORG_LEVEL = """
    Whether the set type is scoped at the organization level (True) or
    project level (False). Org-level sets are shared across all projects
    within the organization."""

    SET_TYPE_METADATA_TAGS = """
    Ordered list of metadata tag keys that define this set type. These tags
    determine how set IDs are generated and grouped. For example,
    ["user_id", "session_id"] means sets are grouped by user and session."""

    SET_TYPE_NAME = """
    Optional human-readable name for the set type."""

    SET_TYPE_DESCRIPTION = """
    Optional description of the set type's purpose."""

    SET_TYPES_LIST = """
    List of semantic set types."""

    SET_METADATA = """
    Optional metadata key-value pairs used to filter or identify semantic sets."""

    SETS_LIST = """
    List of semantic sets."""

    SET_EMBEDDER_NAME = """
    Optional embedder name override for this semantic set. If not specified,
    the default embedder is used."""

    SET_LLM_NAME = """
    Optional language model name override for this semantic set. If not
    specified, the default language model is used."""

    # --- Semantic Category Fields ---

    CATEGORY_ID = """
    Unique identifier for the semantic category."""

    CATEGORY_NAME = """
    Human-readable name for the category. Categories group related features
    together for extraction and organization."""

    CATEGORY_PROMPT = """
    The prompt template used for extracting features in this category.
    This drives the LLM's feature extraction behavior."""

    CATEGORY_DESCRIPTION = """
    Optional human-readable description of the category's purpose."""

    CATEGORY_ORIGIN_TYPE = """
    The origin type of the category: 'set_id' for local categories or
    'set_type' for inherited template categories."""

    CATEGORY_ORIGIN_ID = """
    The identifier of the origin (set_id or set_type_id) where the category
    was defined."""

    CATEGORY_INHERITED = """
    Whether this category is inherited from a set type template."""

    CATEGORIES_LIST = """
    List of semantic categories."""

    CATEGORY_SET_IDS = """
    List of set IDs associated with a category."""

    # --- Semantic Tag Fields ---

    TAG_ID = """
    Unique identifier for the semantic tag."""

    TAG_NAME = """
    Human-readable name for the tag. Tags represent specific types of
    features within a category."""

    TAG_DESCRIPTION = """
    Human-readable description of what this tag represents and when
    features should be tagged with it."""


class Examples:
    """Common examples for API fields."""

    ORG_ID: ClassVar[list[str]] = ["MemVerge", "AI_Labs"]
    PROJECT_ID: ClassVar[list[str]] = ["memmachine", "research123", "qa_pipeline"]
    PROJECT_DESCRIPTION: ClassVar[list[str]] = [
        "Test project for RAG pipeline",
        "Production semantic search index",
    ]
    RERANKER: ClassVar[list[str]] = ["bge-reranker-large", "my-custom-reranker"]
    EMBEDDER: ClassVar[list[str]] = ["bge-base-en", "my-embedder"]
    TOP_K: ClassVar[list[int]] = [5, 10, 20]
    EXPAND_CONTEXT: ClassVar[list[int]] = [0, 3, 6]
    SCORE_THRESHOLD: ClassVar[list[float | None]] = [0.0, 0.5, None]
    QUERY: ClassVar[list[str]] = [
        "What was the user's last conversation about finance?"
    ]
    FILTER_MEM: ClassVar[list[str]] = [
        "metadata.user_id=123 AND metadata.session_id=abc",
    ]
    MEMORY_TYPES: ClassVar[list[list[str]]] = [["episodic", "semantic"]]
    MEMORY_TYPE_SINGLE: ClassVar[list[str]] = ["episodic", "semantic"]
    PAGE_SIZE: ClassVar[list[int]] = [50, 100]
    PAGE_NUM: ClassVar[list[int]] = [0, 1, 5, 10]
    EPISODIC_ID: ClassVar[list[str]] = ["123", "345"]
    EPISODIC_IDS: ClassVar[list[list[str]]] = [["123", "345"], ["23"]]
    SEMANTIC_ID: ClassVar[list[str]] = ["12", "23"]
    SEMANTIC_IDS: ClassVar[list[list[str]]] = [["123", "345"], ["23"]]
    SEARCH_RESULT_STATUS: ClassVar[list[int]] = [0]
    SERVER_VERSION: ClassVar[list[str]] = ["0.1.2", "0.2.0"]
    CLIENT_VERSION: ClassVar[list[str]] = ["0.1.2", "0.2.0"]
    AGENT_MODE: ClassVar[list[bool]] = [True, False]


class RouterDoc:
    """Common descriptions for API routers."""

    CREATE_PROJECT = """
    Create a new project.

    This endpoint creates a project under the specified organization using the
    provided identifiers and configuration. Both `org_id` and `project_id`
    follow the rules: no slashes; only letters, numbers, underscores,
    hyphens, colon, and Unicode characters.

    Each project acts as an isolated memory namespace. All memories (episodes)
    inserted into a project belong exclusively to that project. Queries,
    listings, and any background operations such as memory summarization or
    knowledge extraction only access data within the same project. No
    cross-project memory access is allowed.

    If a project with the same ID already exists within the organization,
    the request will fail with an error.

    Returns the fully resolved project record, including configuration defaults
    applied by the system.
    """

    GET_PROJECT = """
    Retrieve a project.

    Returns the project identified by `org_id` and `project_id`, following
    the same rules as project creation.

    Each project acts as an isolated memory namespace. Queries and operations
    only access memories (episodes) stored within this project. No data from
    other projects is visible or included in any background processing, such as
    memory summarization or knowledge extraction.

    The response includes the project's description and effective configuration.
    If the project does not exist, a not-found error is returned.
    """

    GET_EPISODE_COUNT = """
    Retrieve the episode count for a project.

    An *episode* is the minimal unit of memory stored in the MemMachine system.
    In most cases, a single episode corresponds to one message or interaction
    from a user. Episodes are appended as the project accumulates conversational
    or operational data.

    This endpoint returns the total number of episodes currently recorded for
    the specified project. If the project does not exist, a not-found error is
    returned.
    """

    LIST_PROJECTS = """
    List all projects.

    Returns a list of all projects accessible within the system. Each entry
    contains the project's organization ID and project ID. Identifiers follow
    the standard rules: no slashes; only letters, numbers, underscores,
    hyphens, colon, and Unicode characters.

    Projects are isolated memory namespaces. Memories (episodes) belong
    exclusively to their project. All project operations, including queries and
    any background processes (e.g., memory summarization or knowledge
    extraction), only operate within the project's own data. No cross-project
    access is allowed.
    """

    DELETE_PROJECT = """
    Delete a project.

    Deletes the specified project identified by `org_id` and `project_id`,
    following the same rules as project creation.

    This operation removes the project and all associated memories (episodes)
    permanently from the system. It cannot be undone.

    If the project does not exist, a not-found error is returned.
    """

    ADD_MEMORIES = """
    Add memory messages to a project.

    The `types` field in the request specifies which memory types to add to:
    - If `types` is empty or not provided, memories are added to all types (episodic and semantic)
    - If `types` only contains `"episodic"`, memories are added only to Episodic memory
    - If `types` only contains `"semantic"`, memories are added only to Semantic memory
    - If `types` contains both, memories are added to both types

    Each memory message represents a discrete piece of information to be stored
    in the project's memory system. Messages can include content, metadata,
    timestamps, and other contextual details.

    The producer field indicates who created the message, while the
    produced_for field specifies the intended recipient. These fields help
    provide context for the memory and if provided should be user-friendly names.

    The endpoint accepts a batch of messages to be added in a single request.
    """

    SEARCH_MEMORIES = """
    Search memories within a project.

    System returns the top K relevant memories matching the natural language query.
    The result is sorted by timestamp to help with context.

    The filter field allows for filtering based on metadata key-value pairs.
    The set_metadata field scopes semantic memories to matching semantic sets.
    The types field allows specifying which memory types to include in the search.
    """

    LIST_MEMORIES = """
    List memories within a project.

    System returns a paginated list of memories stored in the project.
    The page_size and page_num fields control pagination.

    The filter field allows for filtering based on metadata key-value pairs.
    The set_metadata field scopes semantic memories to matching semantic sets.
    The type field allows specifying which memory type to list.
    """

    DELETE_EPISODIC_MEMORY = """
    Delete episodic memories from a project.

    This operation permanently removes one or more episodic memories from the
    specified project. You may provide either `episodic_id` to delete a single
    memory or `episodic_ids` to delete multiple memories in one request.
    This action cannot be undone.

    If any of the specified episodic memories do not exist, a not-found error
    is returned for those entries.
    """

    DELETE_SEMANTIC_MEMORY = """
    Delete semantic memories from a project.

    This operation permanently removes one or more semantic memories from the
    specified project. You may provide either `semantic_id` to delete a single
    memory or `semantic_ids` to delete multiple memories in one request.
    This action cannot be undone.

    If any of the specified semantic memories do not exist, a not-found error
    is returned for those entries.
    """

    METRICS_PROMETHEUS = """
    Expose Prometheus metrics."""

    HEALTH_CHECK = """
    Health check endpoint to verify server is running."""

    # --- Configuration API Router Docs ---

    CONFIG_SECURITY_WARNING = """
    **Security Warning**: Configuration changes made via the API are persisted
    to the configuration file in plain text, including API keys and credentials.
    Only use the configuration API in protected environments where the
    configuration file is secured.

    To avoid storing secrets in plain text, you can use environment variable
    references in your API requests (e.g., `"api_key": "$OPENAI_API_KEY"`).
    The server will resolve these at runtime while preserving the reference
    in the configuration file.
    """

    GET_CONFIG = """
    Get current configuration and resource status.

    Returns a comprehensive view of all configured resources in the system,
    including embedders, language models, rerankers, and databases. Each
    resource includes its current status (ready, failed, or pending) and
    any error messages if initialization failed.

    This endpoint is useful for:
    - Monitoring the health of configured resources
    - Diagnosing startup failures when resources are unavailable
    - Verifying that runtime configuration changes have been applied

    Resources may be in one of three states:
    - `ready`: The resource is initialized and available for use
    - `failed`: The resource failed to initialize (error details included)
    - `pending`: The resource is configured but not yet initialized
    """

    GET_RESOURCES = """
    Get status of all configured resources.

    Returns the status of all embedders, language models, rerankers, and
    databases configured in the system. This is a subset of the full
    configuration response, focused specifically on resource health.

    Each resource entry includes:
    - `name`: The unique identifier for the resource
    - `provider`: The type of provider (e.g., 'openai', 'amazon-bedrock')
    - `status`: Current state ('ready', 'failed', or 'pending')
    - `error`: Error message if the resource failed to initialize

    Use this endpoint to quickly check which resources are available before
    making API calls that depend on specific embedders or models.
    """

    ADD_EMBEDDER = """
    Add a new embedder configuration at runtime.

    Creates a new embedder with the specified configuration and attempts to
    initialize it immediately. This allows adding embedders without restarting
    the server. The configuration is persisted to the configuration file.

    **Security Warning**: API keys and credentials are stored in plain text in
    the configuration file. Only use this API in protected environments. To
    avoid storing secrets in plain text, use environment variable references
    (e.g., `"api_key": "$OPENAI_API_KEY"`) which will be resolved at runtime.

    Supported providers:
    - `openai`: OpenAI embedding models (requires api_key, model)
    - `amazon-bedrock`: AWS Bedrock embedding models (requires region, model_id)
    - `sentence-transformer`: Local sentence transformer models (requires model)

    The response indicates whether the embedder was successfully initialized:
    - `success: true, status: ready`: Embedder is available for use
    - `success: false, status: failed`: Initialization failed (check error field)

    If initialization fails, the embedder configuration is still stored and can
    be retried later using the retry endpoint when the underlying service
    becomes available.

    Returns 422 if the provider type or configuration is invalid.
    """

    ADD_LANGUAGE_MODEL = """
    Add a new language model configuration at runtime.

    Creates a new language model with the specified configuration and attempts
    to initialize it immediately. This allows adding models without restarting
    the server. The configuration is persisted to the configuration file.

    **Security Warning**: API keys and credentials are stored in plain text in
    the configuration file. Only use this API in protected environments. To
    avoid storing secrets in plain text, use environment variable references
    (e.g., `"api_key": "$OPENAI_API_KEY"`) which will be resolved at runtime.

    Supported providers:
    - `openai-responses`: OpenAI models using the Responses API (requires api_key, model)
    - `openai-chat-completions`: OpenAI models using Chat Completions API (requires api_key, model)
    - `amazon-bedrock`: AWS Bedrock models (requires region, model_id)

    The response indicates whether the model was successfully initialized:
    - `success: true, status: ready`: Model is available for use
    - `success: false, status: failed`: Initialization failed (check error field)

    If initialization fails, the model configuration is still stored and can
    be retried later using the retry endpoint when the underlying service
    becomes available.

    Returns 422 if the provider type or configuration is invalid.
    """

    DELETE_EMBEDDER = """
    Remove an embedder configuration.

    Permanently removes the specified embedder from the system. This removes
    both the cached instance and the configuration, so the embedder cannot
    be used until it is added again. The change is persisted to the
    configuration file.

    This operation is useful for:
    - Removing embedders that are no longer needed
    - Clearing failed embedders before adding a corrected configuration
    - Freeing resources used by unused embedders

    Returns 404 if the embedder does not exist.
    """

    DELETE_LANGUAGE_MODEL = """
    Remove a language model configuration.

    Permanently removes the specified language model from the system. This
    removes both the cached instance and the configuration, so the model
    cannot be used until it is added again. The change is persisted to the
    configuration file.

    This operation is useful for:
    - Removing models that are no longer needed
    - Clearing failed models before adding a corrected configuration
    - Freeing resources used by unused models

    Returns 404 if the language model does not exist.
    """

    RETRY_EMBEDDER = """
    Retry building a failed embedder.

    Attempts to reinitialize an embedder that previously failed to build.
    This is useful when the underlying service (e.g., OpenAI API, AWS Bedrock)
    was temporarily unavailable during initial startup or configuration.

    The retry clears any previous error state and attempts a fresh build.
    If successful, the embedder becomes available for use immediately.

    The response indicates the result:
    - `success: true, status: ready`: Embedder is now available
    - `success: false, status: failed`: Still failing (check error field)

    Returns 404 if the embedder is not configured.
    """

    RETRY_LANGUAGE_MODEL = """
    Retry building a failed language model.

    Attempts to reinitialize a language model that previously failed to build.
    This is useful when the underlying service (e.g., OpenAI API, AWS Bedrock)
    was temporarily unavailable during initial startup or configuration.

    The retry clears any previous error state and attempts a fresh build.
    If successful, the model becomes available for use immediately.

    The response indicates the result:
    - `success: true, status: ready`: Model is now available
    - `success: false, status: failed`: Still failing (check error field)

    Returns 404 if the language model is not configured.
    """

    UPDATE_MEMORY_CONFIG = """
    Update the episodic and/or semantic memory configuration.

    This endpoint allows updating the memory configuration at runtime after
    resources (embedders, language models, rerankers) have been added or
    changed. Without calling this endpoint, newly added resources will not
    be used by the memory systems.

    Both `episodic_memory` and `semantic_memory` are optional. Supply one
    or both depending on which sections need updating. Within each section,
    only the fields you supply are modified; omitted fields retain their
    current values.

    **Typical workflow:**
    1. Add a new resource via `POST /config/resources/embedders` or
       `POST /config/resources/language_models`
    2. Call this endpoint to point the memory configuration at the new resource
    3. The configuration is persisted to the configuration file

    **Example - update episodic long-term memory to use a new embedder:**
    ```json
    {
      "episodic_memory": {
        "long_term_memory": {
          "embedder": "my-new-embedder"
        }
      }
    }
    ```

    **Example - update semantic memory to use a new LLM and embedder:**
    ```json
    {
      "semantic_memory": {
        "llm_model": "my-new-model",
        "embedding_model": "my-new-embedder"
      }
    }
    ```

    Returns 400 if no updates are supplied (both sections are null or empty).
    """

    RETRY_RERANKER = """
    Retry building a failed reranker.

    Attempts to reinitialize a reranker that previously failed to build.
    This is useful when the underlying service or dependencies were
    temporarily unavailable during initial startup.

    The retry clears any previous error state and attempts a fresh build.
    If successful, the reranker becomes available for use immediately.

    The response indicates the result:
    - `success: true, status: ready`: Reranker is now available
    - `success: false, status: failed`: Still failing (check error field)

    Returns 404 if the reranker is not configured.
    """

    ADD_FEATURE = """
    Add a semantic feature to a project.

    Creates a new semantic feature with the specified category, tag, feature name,
    and value. Optional metadata and episode citations can be attached.

    Returns the unique identifier of the created feature.
    """

    GET_FEATURE = """
    Get a semantic feature by ID.

    Retrieves a specific semantic feature using its unique identifier.
    Optionally includes episode citations if `load_citations` is true.

    Returns 404 if the feature does not exist.
    """

    UPDATE_FEATURE = """
    Update a semantic feature.

    Updates an existing semantic feature with new values. Only the fields
    provided in the request are updated; omitted fields retain their current
    values.

    Returns 404 if the feature does not exist.
    """

    UPDATE_LTM_CONFIG = """
    Update long-term memory configuration.

    This endpoint updates the long-term memory configuration at runtime.
    Only the fields you supply are modified; omitted fields retain their
    current values.

    The configuration includes:
    - embedder: The embedder resource to use for creating embeddings
    - reranker: The reranker resource to use for search result reranking
    - vector_graph_store: The database for storing long-term memories
    - enabled: Whether long-term memory is enabled
    """

    UPDATE_STM_CONFIG = """
    Update short-term memory configuration.

    This endpoint updates the short-term memory configuration at runtime.
    Only the fields you supply are modified; omitted fields retain their
    current values.

    The configuration includes:
    - llm_model: The language model to use for summarization
    - message_capacity: Maximum message capacity in characters
    - enabled: Whether short-term memory is enabled
    """

    GET_EPISODIC_CONFIG = """
    Get episodic memory configuration.

    Returns the current episodic memory configuration including both
    long-term and short-term memory settings, and the overall enabled status.
    """

    UPDATE_EPISODIC_CONFIG = """
    Update episodic memory configuration.

    This endpoint updates the episodic memory configuration at runtime,
    including both long-term and short-term memory settings.
    Only the fields you supply are modified; omitted fields retain their
    current values.

    The configuration includes:
    - long_term_memory: Long-term memory settings (embedder, reranker, vector_graph_store)
    - short_term_memory: Short-term memory settings (llm_model, message_capacity)
    - long_term_memory_enabled: Whether long-term memory is enabled
    - short_term_memory_enabled: Whether short-term memory is enabled
    - enabled: Whether episodic memory is enabled overall
    """

    GET_LTM_CONFIG = """
    Get long-term memory configuration.

    Returns the current long-term memory configuration including embedder,
    reranker, vector graph store, and enabled status.
    """

    GET_STM_CONFIG = """
    Get short-term memory configuration.

    Returns the current short-term memory configuration including LLM model,
    message capacity, and enabled status.
    """

    GET_SEMANTIC_CONFIG = """
    Get semantic memory configuration.

    Returns the current semantic memory configuration including database,
    LLM model, embedding model, and enabled status.
    """

    UPDATE_SEMANTIC_CONFIG = """
    Update semantic memory configuration.

    This endpoint updates the semantic memory configuration at runtime.
    Only the fields you supply are modified; omitted fields retain their
    current values.

    The configuration includes:
    - enabled: Whether semantic memory is enabled
    - database: The database resource to use for storing semantic memories
    - llm_model: The language model to use for feature extraction
    - embedding_model: The embedder to use for semantic similarity
    - ingestion_trigger_messages: Number of messages before triggering ingestion
    - ingestion_trigger_age_seconds: Age threshold for triggering ingestion
    """

    # --- Semantic Set Type API Router Docs ---

    CREATE_SEMANTIC_SET_TYPE = """
    Create a new semantic set type.

    A set type defines a template for grouping semantic features based on
    metadata tags. For example, a set type with tags ["user_id"] creates
    user-scoped feature groups.

    Set types can be organization-level (shared across projects) or
    project-level (scoped to a single project).

    Returns the unique identifier of the created set type.
    """

    LIST_SEMANTIC_SET_TYPES = """
    List all semantic set types.

    Returns all set types defined for the organization. Each set type
    includes its metadata tag configuration and scope level.
    """

    DELETE_SEMANTIC_SET_TYPE = """
    Delete a semantic set type.

    Permanently removes the specified set type. This does not delete
    the sets or features associated with the set type.

    Returns 404 if the set type does not exist.
    """

    GET_SEMANTIC_SET_ID = """
    Get or create a semantic set ID.

    Returns the set ID for the given set type configuration and optional
    metadata. If the set does not exist, it is created.

    The set ID is deterministically generated based on:
    - Organization and project scope
    - Set type (org-level or project-level)
    - Metadata tags and values
    """

    LIST_SEMANTIC_SET_IDS = """
    List all semantic sets.

    Returns all sets for the organization/project context. Can be filtered
    by metadata values.
    """

    CONFIGURE_SEMANTIC_SET = """
    Configure a semantic set.

    Updates the embedder and/or language model used for a specific set.
    This allows customizing the models used for feature extraction and
    embedding on a per-set basis.

    If not configured, sets inherit the default embedder and language model.
    """

    # --- Semantic Category API Router Docs ---

    GET_SEMANTIC_CATEGORY = """
    Get a semantic category by ID.

    Retrieves a specific category using its unique identifier.
    Returns the category name, prompt, and description.

    Returns 404 if the category does not exist.
    """

    ADD_SEMANTIC_CATEGORY = """
    Add a semantic category to a set.

    Creates a new category within a specific set. Categories define how
    features are extracted and organized. Each category has:
    - A name for identification
    - A prompt that drives LLM feature extraction
    - An optional description

    Returns the unique identifier of the created category.
    """

    ADD_SEMANTIC_CATEGORY_TEMPLATE = """
    Add a semantic category template to a set type.

    Creates a category that will be inherited by all sets mapped to this
    set type. Template categories provide default extraction behavior
    that can be customized or disabled per-set.

    Returns the unique identifier of the created category template.
    """

    LIST_SEMANTIC_CATEGORY_TEMPLATES = """
    List semantic category templates for a set type.

    Returns all categories defined on a set type. These categories
    are inherited by sets mapped to the set type.
    """

    DISABLE_SEMANTIC_CATEGORY = """
    Disable a semantic category for a set.

    Prevents a category from being used for feature extraction on a
    specific set. This is useful for disabling inherited categories
    without deleting them.

    The category remains available for other sets.
    """

    GET_SEMANTIC_CATEGORY_SET_IDS = """
    Get set IDs associated with a category.

    Returns all set IDs that have features in the specified category.
    Useful for understanding the scope of a category's usage.
    """

    DELETE_SEMANTIC_CATEGORY = """
    Delete a semantic category.

    Permanently removes the category and all associated tags and features.
    This is a destructive operation that cannot be undone.

    Returns 404 if the category does not exist.
    """

    # --- Semantic Tag API Router Docs ---

    ADD_SEMANTIC_TAG = """
    Add a tag to a semantic category.

    Creates a new tag within a category. Tags represent specific types
    of features that can be extracted. Each tag has:
    - A name for identification
    - A description explaining what the tag represents

    Returns the unique identifier of the created tag.
    """

    DELETE_SEMANTIC_TAG = """
    Delete a semantic tag.

    Permanently removes the tag from its category. Features tagged with
    this tag may lose their tag association.

    Returns 404 if the tag does not exist.
    """

    # --- Episodic Memory Configuration Endpoints ---

    GET_EPISODIC_MEMORY_CONFIG = """
    Get episodic memory configuration for a project.

    Returns the current episodic memory configuration for the specified
    project, including whether episodic memory, long-term memory, and
    short-term memory are enabled.

    Returns 404 if the project does not exist.
    """

    CONFIGURE_EPISODIC_MEMORY = """
    Configure episodic memory for a project.

    Allows enabling or disabling episodic memory components for a specific
    project session. You can independently control:

    - **enabled**: Master switch for all episodic memory. When disabled,
      no episodic memories will be stored or retrieved.
    - **long_term_memory_enabled**: Controls long-term memory storage.
      Long-term memory provides persistent storage with summarization
      and vector search capabilities.
    - **short_term_memory_enabled**: Controls short-term memory storage.
      Short-term memory maintains recent context for the current session.

    Only provided (non-null) values are updated. Fields not specified
    in the request retain their current values.

    Returns 404 if the project does not exist.
    """

    GET_SHORT_TERM_MEMORY_CONFIG = """
    Get short-term memory configuration for a project.

    Returns whether short-term episodic memory is enabled for the specified
    project.

    Returns 404 if the project does not exist.
    """

    CONFIGURE_SHORT_TERM_MEMORY = """
    Configure short-term memory for a project.

    Allows enabling or disabling short-term episodic memory for a specific
    project session. Short-term memory maintains recent context for the
    current session.

    Returns 404 if the project does not exist.
    """

    GET_LONG_TERM_MEMORY_CONFIG = """
    Get long-term memory configuration for a project.

    Returns whether long-term episodic memory is enabled for the specified
    project.

    Returns 404 if the project does not exist.
    """

    CONFIGURE_LONG_TERM_MEMORY = """
    Configure long-term memory for a project.

    Allows enabling or disabling long-term episodic memory for a specific
    project session. Long-term memory provides persistent storage with
    summarization and vector search capabilities.

    Returns 404 if the project does not exist.
    """
