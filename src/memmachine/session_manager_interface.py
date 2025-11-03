from abc import ABC, abstractmethod



class SessionDataManager(ABC):
    """
    Interface for managing session data, including session configurations and
    short-term memory.
    """

    @classmethod
    async def close(self):
        """
        Closes the database connection.
        """
        raise NotImplementedError

    @abstractmethod
    async def create_tables(self):
        """
        Creates the necessary tables in the database.
        """
        raise NotImplementedError

    @abstractmethod
    async def create_new_session(
        self, session_key: str, configuration: dict, param: 'EpisodicMemoryParams', description: str, metadata: dict
    ):
        """
        Creates a new session entry in the database.

        Args:
            session_key: The unique identifier for the session.
            configuration: A dictionary containing the session's configuration.
            description: A brief description of the session.
            metadata: A dictionary for user-defined metadata.

        Raises:
            ValueError: If a session with the given session_key already exists.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_session(self, session_key: str):
        """
        Deletes a session entry from the database.

        Args:
            session_key: The unique identifier of the session to delete.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_session_info(self, session_key: str) -> tuple[dict, str, dict, 'EpisodicMemoryParams']:
        """
        Retrieves the configuration, description, and metadata for a given
        session.

        Args:
            session_key: The unique identifier of the session.

        Returns:
            A tuple containing the configuration dictionary, description string,
            metadata dictionary and the EpisodicMemoryParams.

        Raises:
            ValueError: If the session with the given session_key does not exist.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_sessions(self, filter: dict[str, str] | None = None) -> list[str]:
        """
        Retrieves a list of all session keys from the database.

        Returns:
            A list of session keys.
        """
        raise NotImplementedError

    @abstractmethod
    async def save_short_term_memory(self, session_key: str, summary: str, last_seq, episode_num: int):
        """
        Saves or updates the short-term memory data for a session.

        Args:
            session_key: The unique identifier for the session.
            summary: The summary of the short-term memory.
            episode_num: The number of episodes in the short-term memory.

        Raises:
            ValueError: If the session with the given session_key does not exist.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_short_term_memory(self, session_key: str) -> tuple[str, int, int]:
        """
        Retrieves the short-term memory data for a session.

        Args:
            session_key: The unique identifier for the session
        Returns:
            A tuple containing the summary string and the number of episodes and the last sequence number.
        """
        raise NotImplementedError
