"""Custom exceptions for MemMachine."""


class MemMachineError(RuntimeError):
    """Base class for MemMachine errors."""


class InvalidArgumentError(MemMachineError):
    """Error for invalid arguments."""


class ResourceNotFoundError(InvalidArgumentError):
    """Error when a specified resource is not found."""


class RerankerNotFoundError(ResourceNotFoundError):
    """Error when a specified reranker is not found."""

    def __init__(self, reranker_name: str) -> None:
        """Initialize with the name of the missing reranker."""
        self.reranker_name = reranker_name
        super().__init__(
            f"Reranker '{reranker_name}' is not defined in the configuration."
        )


class EmbedderNotFoundError(ResourceNotFoundError):
    """Error when a specified embedder is not found."""

    def __init__(self, embedder_name: str) -> None:
        """Initialize with the name of the missing embedder."""
        self.embedder_name = embedder_name
        super().__init__(
            f"Embedder '{embedder_name}' is not defined in the configuration."
        )


class ConfigurationError(MemMachineError):
    """Error related to system configuration."""


class DefaultRerankerNotConfiguredError(ConfigurationError):
    """Error when default reranker is missing."""

    def __init__(self) -> None:
        """Initialize the error."""
        super().__init__("No default reranker is configured.")


class DefaultEmbedderNotConfiguredError(ConfigurationError):
    """Error when default embedder is missing."""

    def __init__(self) -> None:
        """Initialize the error."""
        super().__init__("No default embedder is configured.")


class SessionAlreadyExistsError(MemMachineError):
    """Error when trying to create a session that already exists."""

    def __init__(self, session_key: str) -> None:
        """Initialize with the session key that already exists."""
        self.session_key = session_key
        super().__init__(f"Session '{session_key}' already exists.")


class SessionNotFoundError(MemMachineError):
    """Error when trying to retrieve a session."""

    def __init__(self, session_key: str) -> None:
        """Initialize with the session key that does not exist."""
        self.session_key = session_key
        super().__init__(f"Session '{session_key}' does not exist.")


class SessionInUseError(MemMachineError):
    """Error when trying to delete a session that is currently in use."""

    def __init__(self, session_key: str, ref_count: int = 0) -> None:
        """Initialize with the session key that is in use."""
        self.session_key = session_key
        msg = f"Session '{session_key}' is in use and can't be deleted."
        if ref_count > 0:
            msg += f" Reference count: {ref_count}."
        super().__init__(msg)


class ShortTermMemoryClosedError(MemMachineError):
    """Error when trying to access closed short-term memory."""

    def __init__(self, session_key: str) -> None:
        """Initialize with the session key of the closed short-term memory."""
        self.session_key = session_key
        super().__init__(f"Short-term memory for session '{session_key}' is closed.")


class InvalidPasswordError(MemMachineError):
    """Error for invalid password scenarios."""


class Neo4JConfigurationError(MemMachineError):
    """Error related to Neo4J configuration."""


class SQLConfigurationError(MemMachineError):
    """Error related to SQL configuration."""


class InvalidLanguageModelError(MemMachineError):
    """Exception raised for invalid language model."""


class InvalidEmbedderError(MemMachineError):
    """Exception raised for invalid embedder."""


class InvalidRerankerError(MemMachineError):
    """Exception raised for invalid reranker."""


class EpisodicMemoryManagerClosedError(MemMachineError):
    """Exception raised when operating on a closed EpisodicMemory instance."""

    def __init__(self) -> None:
        """Initialize the error."""
        super().__init__("The EpisodicMemoryManager is closed and cannot be used.")
