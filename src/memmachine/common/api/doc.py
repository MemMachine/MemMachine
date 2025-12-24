"""API documentation strings for MemMachine server v2."""

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
    The types field allows specifying which memory types to include in the search.
    """

    LIST_MEMORIES = """
    List memories within a project.

    System returns a paginated list of memories stored in the project.
    The page_size and page_num fields control pagination.

    The filter field allows for filtering based on metadata key-value pairs.
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
