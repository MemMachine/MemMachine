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
