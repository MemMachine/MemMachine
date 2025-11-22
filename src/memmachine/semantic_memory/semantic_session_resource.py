"""Request/response models for semantic memory session resources."""

from enum import Enum
from typing import Final, Protocol, runtime_checkable

from memmachine.semantic_memory.semantic_model import Resources, SetIdT


class IsolationType(Enum):
    """Isolation scopes supported when mapping activity to semantic-memory set_ids."""

    USER = "user_profile"
    ROLE = "role_profile"
    SESSION = "session"


ALL_MEMORY_TYPES: Final[list[IsolationType]] = list(IsolationType)


@runtime_checkable
class SessionData(Protocol):
    """Protocol exposing the identifiers used to derive set_ids."""

    @property
    def user_profile_id(self) -> SetIdT | None:
        raise NotImplementedError

    @property
    def session_id(self) -> SetIdT | None:
        raise NotImplementedError

    @property
    def role_profile_id(self) -> SetIdT | None:
        raise NotImplementedError


@runtime_checkable
class SessionIdIsolationTypeChecker(Protocol):
    """Protocol for determining the isolation type associated with a set_id."""

    def set_id_isolation_type(self, set_id: SetIdT) -> IsolationType:
        raise NotImplementedError


class SessionIdManager:
    """Generates namespaced set_ids and reports which isolation scope they belong to."""

    def __init__(self) -> None:
        """Initialize the session id manager."""

    _SESSION_ID_PREFIX: Final[str] = "mem_session_"
    _USER_ID_PREFIX: Final[str] = "mem_user_"
    _ROLE_ID_PREFIX: Final[str] = "mem_role_"

    def generate_session_data(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        role_id: str | None = None,
    ) -> SessionData:
        class _SessionDataImpl:
            """Lightweight `SessionData` implementation backed by generated set_ids."""

            def __init__(
                self,
                *,
                _user_profile_id: str | None,
                _role_profile_id: str | None,
                _session_id: str | None,
            ) -> None:
                """Capture the identifiers used to scope memory resources."""
                self._user_id: str | None = _user_profile_id
                self._role_id: str | None = _role_profile_id
                self._session_id: str | None = _session_id

            @property
            def user_profile_id(self) -> str | None:
                return self._user_id

            @property
            def role_profile_id(self) -> str | None:
                return self._role_id

            @property
            def session_id(self) -> str | None:
                return self._session_id

        return _SessionDataImpl(
            _user_profile_id=self._USER_ID_PREFIX + user_id if user_id else None,
            _session_id=self._SESSION_ID_PREFIX + session_id if session_id else None,
            _role_profile_id=self._ROLE_ID_PREFIX + role_id if role_id else None,
        )

    def set_id_isolation_type(self, set_id: SetIdT) -> IsolationType:
        if self.is_session_id(set_id):
            return IsolationType.SESSION
        if self.is_producer_id(set_id):
            return IsolationType.USER
        if self.is_role_id(set_id):
            return IsolationType.ROLE
        raise ValueError(f"Invalid id: {set_id}")

    def is_session_id(self, _id: str) -> bool:
        return _id.startswith(self._SESSION_ID_PREFIX)

    def is_producer_id(self, _id: str) -> bool:
        return _id.startswith(self._USER_ID_PREFIX)

    def is_role_id(self, _id: str) -> bool:
        return _id.startswith(self._ROLE_ID_PREFIX)


class SessionResourceRetriever:
    """Resolves the `Resources` bundle for a set_id, falling back to isolation defaults."""

    def __init__(
        self,
        session_id_manager: SessionIdIsolationTypeChecker,
        default_resources: dict[IsolationType, Resources],
    ) -> None:
        """Initialize the retriever with isolation defaults and a session checker."""
        self._session_id_manager = session_id_manager
        self._default_resources = default_resources

    def get_resources(self, set_id: str) -> Resources:
        custom_resources = self._get_set_id_resources(set_id)
        if custom_resources is not None:
            return custom_resources

        set_id_type = self._session_id_manager.set_id_isolation_type(set_id)

        return self._default_resources[set_id_type]

    @staticmethod
    def _get_set_id_resources(_: str) -> Resources | None:
        # TODO: Load set_id custom resources
        return None
